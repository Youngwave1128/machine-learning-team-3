"""
Hybrid Recommendation System: CF + CBF(BERT) + XGBoost (End-to-End)

[프로젝트 목적]
- (cluster_id, rent_for)로 정의된 '컨텍스트(Context)'별로 드레스 아이템 Top-N 추천

[핵심 설계 (Hybrid Pipeline)]
1.  (CF - rating, fit): 미리 계산된 평점/핏 점수를 사용 (cf_model_final.csv)
2.  (CBF) 콘텐츠 기반 필터링: BERT로 아이템 텍스트(이름, 설명)를 임베딩
3.  (XGBoost) 랭킹 모델: 1번(CF)과 2번(CBF)의 점수를 피처(feature)로 입력받아,
    사용자의 실제 '평점(rating)'을 예측하도록 학습하는 랭킹 모델
"""

import os
import sys
import json
import math
import time
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sentence_transformers import SentenceTransformer  # CBF용 BERT
from xgboost import XGBRegressor                       # 하이브리드 랭킹(메타) 모델



# 설정 (입출력 파일/하이퍼파라미터)

FILE_TRAIN_CLU = "renttherunway_clustered.csv"  # 2단계 출력 (클러스터+context 포함)
FILE_TEST_CLU = "modcloth_clustered.csv"        # 2단계 출력 (테스트)
FILE_CF_FINAL = "cf_model_final.csv"            # 3단계 출력 (CF rating/fit/n_users)
FILE_ARTICLES = "articles.csv"                  # 아이템 메타 (prod_name, detail_desc 등)
TOPK_EVAL = 10                                  # 평가 K
RANDOM_STATE = 42                               # 재현성 (Reproducibility)

# XGBoost 랭킹 모델이 사용할 4가지 핵심 피처
FEATURE_COLS = ["cf_rating_score", "cf_fit_score", "n_users", "cbf_style_score"]

# 컨텍스트별 CBF 점수 캐시 (중복 계산 방지용 글로벌 변수)
_CBF_CACHE = {}


# -----------------------------
# 1) 데이터 로드 및 컨텍스트 정리
# -----------------------------
def load_and_prepare():
    """
     CSV 파일 로드 및 전처리
    - 모든 데이터의 'context' 형식을 통일 (예: 'cluster_0_rent_date' -> 'cluster_0_date')
    - 'product_code'를 10자리 문자열(0-padding)로 통일
    - 'n_reviews' 컬럼명을 'n_users'로 표준화
    
    Returns:
        tuple: (df_train, df_test, df_cf, df_art) 4개 데이터프레임
    """
    print("--- 1. 데이터 로드 및 전처리 ---")
    dtype_spec = {"product_code": str, "user_id": str}
    df_train = pd.read_csv(FILE_TRAIN_CLU, dtype=dtype_spec)
    df_test = pd.read_csv(FILE_TEST_CLU, dtype=dtype_spec)
    df_cf = pd.read_csv(FILE_CF_FINAL, dtype=dtype_spec)
    df_art = pd.read_csv(FILE_ARTICLES, dtype=dtype_spec, encoding="latin-1")

    for df in (df_train, df_test, df_cf, df_art):
        df.columns = df.columns.str.strip()
        if "product_code" in df.columns:
            df["product_code"] = df["product_code"].astype(str).str.zfill(10)

    # --- Context 생성/보정 ---
    # Train/Test 데이터에 context가 없으면 'cluster_id'와 'rented for'로 생성
    if "context" not in df_train.columns:
        df_train["context"] = 'cluster_' + df_train["cluster_id"].astype(str) + '_' + df_train["rented for"]
    if "context" not in df_test.columns:
        df_test["context"] = 'cluster_' + df_test["cluster_id"].astype(str) + '_' + df_test["rented for"]

    # CF 데이터의 '_rent_' 오타를 '_'로 수정하여 형식 통일
    if "context" in df_cf.columns:
        # 예: cluster_0_rent_date_general → cluster_0_date_general
        df_cf["context"] = df_cf["context"].str.replace("_rent_", "_", regex=False)
        print("[수정] CF context에서 '_rent_' 제거 완료.")
    else:
        # CF 데이터에도 context가 없으면 생성
        df_cf["context"] = 'cluster_' + df_cf["cluster_id"].astype(str) + '_' + df_cf["rented for"]

    # n_users 컬럼 정리 (n_reviews -> n_users)
    if "n_users" not in df_cf.columns and "n_reviews" in df_cf.columns:
        df_cf = df_cf.rename(columns={"n_reviews": "n_users"})
    df_cf["n_users"] = pd.to_numeric(df_cf["n_users"], errors="coerce").fillna(0).astype(int)

    return df_train, df_test, df_cf, df_art


# -----------------------------
# 2) CBF: BERT 스타일 임베딩 구축
# -----------------------------
def build_bert_embeddings(df_train, df_art):
    """
    아이템(드레스)의 텍스트 정보 -> 고차원 벡터(임베딩)로 변환
    - 콘텐츠: prod_name + detail_desc (= style_text)
    - 모델: paraphrase-multilingual-MiniLM-L12-v2 
    
    Args:
        df_train (pd.DataFrame): 학습 데이터 (사용된 아이템 필터링용)
        df_art (pd.DataFrame): 아이템 메타 정보
        
    Returns:
        tuple: (
            df_dress (pd.DataFrame): 임베딩된 드레스 정보,
            bert_matrix (np.ndarray): N x 384 크기의 임베딩 행렬,
            item_index_map (dict): product_code -> bert_matrix 행 인덱스 매핑
        )
    """
  
    print("\n--- 2. CBF: BERT 스타일 모델 구축 ---")
  
    # 2-1) 훈련에 실제 등장한 아이템(used_codes)만 대상으로 하여 연산량 절감
    used_codes = set(df_train["product_code"].unique())
    df_art_filtered = df_art[df_art["product_code"].isin(used_codes)].copy()

    # 2-2) 드레스만 사용 (없으면 전체 사용)
    df_dress = df_art_filtered[df_art_filtered["product_type_name"] == "Dress"].copy()
    if df_dress.empty:
        print("[경고] Dress 타입이 없어 모든 타입 사용")
        df_dress = df_art_filtered.copy()

    # 2-3) 스타일 텍스트 생성 (prod_name + detail_desc)
    df_dress["prod_name"] = df_dress["prod_name"].fillna("no name").astype(str)
    df_dress["detail_desc"] = df_dress.get("detail_desc", pd.Series(index=df_dress.index)).fillna("no description").astype(str)
    df_dress["style_text"] = (df_dress["prod_name"] + " " + df_dress["detail_desc"]).str.strip()
    df_dress = df_dress.drop_duplicates(subset=["product_code"]).reset_index(drop=True)

    # 2-4) BERT 모델 로드 및 임베딩 수행
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    bert_matrix = model.encode(df_dress["style_text"].tolist(), show_progress_bar=True)

    # 2-5) product_code -> bert_matrix의 인덱스(순서)를 매핑하는 딕셔너리 생성
    item_index_map = {code: i for i, code in enumerate(df_dress["product_code"].tolist())}
    return df_dress, bert_matrix, item_index_map


def make_ctx2codes(df_train):
    """
    CBF 사용자 프로필 생성을 위해, 각 'context'별로 유저들이 입었던 'product_code' 리스트를 매핑
    
    Args:
        df_train (pd.DataFrame): 학습 데이터
        
    Returns:
        dict: {'context_name': ['code1', 'code2', ...], ...}
    """
  
    return (
        df_train[["context", "product_code"]]
        .dropna()
        .groupby("context")["product_code"]
        .apply(list)
        .to_dict()
    )


# -----------------------------
# 3) CBF 점수 계산 (핵심 함수)
# -----------------------------
def compute_cbf_for_context(ctx, ctx2codes, df_dress, bert_matrix, item_index_map):
    """
    특정 context의 '스타일 점수' 계산
    - 로직:
        1. 이 context에서 유저들이 입었던 아이템들의 '평균 스타일 벡터' (U_c) 계산
        2. U_c와 '모든 드레스'의 벡터(v_j) 간의 코사인 유사도 점수 계산
    - 최적화: 글로벌 변수 _CBF_CACHE를 사용해 이미 계산된 context는 즉시 반환
    
    Args:
        ctx (str): 컨텍스트 명 (예: 'cluster_5_wedding_black_tie')
        (나머지 파라미터): build_bert_embeddings, make_ctx2codes의 반환값
        
    Returns:
        pd.DataFrame: ['product_code', 'cbf_style_score'] 컬럼을 가진 데이터프레임
    """
  
    # (최적화) 이미 계산된 결과가 캐시에 있으면 바로 반환
    if ctx in _CBF_CACHE:
        return _CBF_CACHE[ctx]

    # 1. 이 context에서 유저들이 입었던 아이템 코드(codes) -> 벡터 인덱스(idxs) 추출
    codes = ctx2codes.get(ctx, [])
    idxs = [item_index_map.get(c) for c in codes if c in item_index_map]

    if not idxs:
        # (예외 처리) 이 context에 과거 착용 내역이 전혀 없는 경우 (Cold Context)
        # 모든 아이템의 스타일 점수를 0점으로 반환
        cbf = pd.DataFrame({
            "product_code": df_dress["product_code"].values,
            "cbf_style_score": np.zeros(len(df_dress), dtype=float)
        })
    else:
        # 2. 유저(컨텍스트) 프로필 벡터 (U_c) 계산: 착용 아이템 벡터들의 평균
        U_c = bert_matrix[idxs].mean(axis=0).reshape(1, -1)
        
        # 3. U_c와 전체 아이템(bert_matrix) 간의 코사인 유사도 계산
        scores = cosine_similarity(U_c, bert_matrix).ravel()
        
        cbf = pd.DataFrame({
            "product_code": df_dress["product_code"].values,
            "cbf_style_score": scores
        })

    _CBF_CACHE[ctx] = cbf  # (최적화) 계산 결과를 캐시에 저장
    return cbf


# -----------------------------
# 4) 학습 피처 구성
# -----------------------------
def build_train_features(df_train, df_cf, df_dress, bert_matrix, item_index_map):
    """
    XGBoost 랭킹 모델 학습용 (X, y) 데이터셋 구성
    - y (정답): 실제 평점 (rating)
    - X (피처): FEATURE_COLS 4가지 (CF 3개 + CBF 1개)
    
    Returns:
        tuple: (
            X (np.ndarray): N x 4 크기의 학습 피처,
            y (np.ndarray): N 크기의 정답 레이블,
            feat (pd.DataFrame): 피처 병합이 완료된 학습 데이터프레임 (디버깅용),
            ctx2codes (dict): CBF 계산에 사용된 context-아이템 매핑 (추천 시 재사용)
        )
    """
    print("\n--- 3. XGBoost 훈련 데이터 구성 ---")
    df_train = df_train.copy()
    df_train["rating"] = pd.to_numeric(df_train["rating"], errors="coerce")
    train_rows = df_train.dropna(subset=["rating", "product_code", "context"])

    # 4-1) CF 피처 3개 병합
    feat = (
        train_rows[["user_id", "product_code", "context", "rating"]]
        .merge(
            df_cf[["context", "product_code", "cf_rating_score", "cf_fit_score", "n_users"]],
            on=["context", "product_code"],
            how="left"
        )
    )
    # 결측치(CF 점수가 없는 아이템)는 0점으로 처리
    feat["cf_rating_score"] = feat["cf_rating_score"].fillna(0.0)
    feat["cf_fit_score"] = feat["cf_fit_score"].fillna(0.0)
    feat["n_users"] = feat["n_users"].fillna(0).astype(int)

    # 4-2) CBF 피처 1개 병합
    # 학습 데이터에 등장하는 모든 context에 대해 3번 함수(compute_cbf_for_context)를 미리 호출
    ctx2codes = make_ctx2codes(df_train)
    cbf_all = []
    for ctx in tqdm(feat["context"].unique(), desc="[Build Feat] CBF per context"):
        cbf_df = compute_cbf_for_context(ctx, ctx2codes, df_dress, bert_matrix, item_index_map)
        cbf_df["context"] = ctx  # 조인 키(context) 추가
        cbf_all.append(cbf_df)
    
    cbf_feat = pd.concat(cbf_all, ignore_index=True)

    feat = feat.merge(cbf_feat, on=["context", "product_code"], how="left")
    feat["cbf_style_score"] = feat["cbf_style_score"].fillna(0.0)

    X = feat[FEATURE_COLS].values
    y = feat["rating"].values
    return X, y, feat, ctx2codes


# -----------------------------
# 5) XGBoost 메타 회귀 학습
# -----------------------------
def train_xgb(X, y, random_state=RANDOM_STATE):
    """
    CF/CBF 피처(X)로 실제 평점(y)을 예측하는 XGBoost 랭킹 모델 학습
    - 조기 종료(early_stopping_rounds)를 사용해 최적의 학습 트리 수를 찾고 과적합 방지
    
    Args:
        X (np.ndarray): N x 4 피처
        y (np.ndarray): N 정답 (평점)
        
    Returns:
        XGBRegressor: 학습 완료된 모델 객체
    """
    print("\n--- 4. XGBoost 모델 학습 ---")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    model = XGBRegressor(
        objective="reg:squarederror",    # 회귀(평점 예측) 문제
        learning_rate=0.1,
        max_depth=5,                     # 트리의 최대 깊이
        n_estimators=1000,               # 최대 1000개 트리 생성
        subsample=0.8,                   # 각 트리마다 80%의 데이터만 사용
        colsample_bytree=0.8,            # 각 트리마다 80%의 피처만 사용
        min_child_weight=3,
        n_jobs=-1,                       # 모든 CPU 코어 사용
        random_state=random_state,
        early_stopping_rounds=50         # 50라운드 동안 검증(eval) 점수 향상이 없으면 조기 종료
    )
    
    # 학습 실행 (검증셋(X_va, y_va) 기준으로 조기 종료 수행)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    # 검증셋 성능 출력
    pred = model.predict(X_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    r2 = float(r2_score(y_va, pred))
    print(f"[METRIC] RMSE={rmse:.4f} | R²={r2:.4f}")
    
    # 피처 중요도 출력
    print("\n[XGB] Feature Importance:")
    imp_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imp_df.to_string(index=False))
    
    return model


# -----------------------------
# 6) 추천 함수
# -----------------------------
def recommend_for_context(ctx, top_n, df_cf, df_dress, model, ctx2codes, bert_matrix, item_index_map):
    """
    (최종 추천) 특정 context에 대해 Top-N 아이템 추천
    - 로직: 4번(학습 데이터 구성)과 동일하게 피처(CF+CBF)를 구성한 뒤,
            5번(학습된 모델)으로 '최종 점수'를 예측하고 정렬
            
    Args:
        ctx (str): 추천 대상 컨텍스트
        top_n (int): 추천할 아이템 개수
        (나머지 파라미터): 이전 단계에서 반환된 객체들
        
    Returns:
        pd.DataFrame: ['product_code', 'prod_name', 'final_score'] 컬럼의 Top-N 추천 목록
    """
    # 1. CF 피처 로드: 이 context에 속한 '모든 아이템 후보군' (df_cf 기반)
    base = df_cf[df_cf["context"] == ctx][["product_code", "cf_rating_score", "cf_fit_score", "n_users"]].copy()
    if base.empty:
        return pd.DataFrame(columns=["product_code", "prod_name", "final_score"])

    # 2. CBF 피처 로드
    cbf_ctx = compute_cbf_for_context(ctx, ctx2codes, df_dress, bert_matrix, item_index_map)
    
    # 3. CF + CBF 피처 병합
    base = base.merge(cbf_ctx, on="product_code", how="left").fillna({"cbf_style_score": 0.0})

    # 4. XGBoost 모델로 '최종 점수' 예측
    X_all = base[FEATURE_COLS].values
    base["final_score"] = model.predict(X_all)
    
    # 5. 아이템 이름(prod_name) 병합 후 Top-N 반환
    base = base.merge(df_dress[["product_code", "prod_name"]], on="product_code", how="left")

    return (
        base.sort_values("final_score", ascending=False)
            .head(top_n)[["product_code", "prod_name", "final_score"]]
            .reset_index(drop=True)
    )


# -----------------------------
# 7) 랭킹 평가
# -----------------------------
def evaluate_topk_on_test(df_test, K, df_cf, df_dress, model, ctx2codes, bert_matrix, item_index_map):
    """
    테스트셋(df_test)을 기준으로 랭킹 성능(Recall@K, MRR@K) 평가
    
    Args:
        df_test (pd.DataFrame): 유저의 (context, true_code) 정답셋
        K (int): 평가 기준 (예: 10)
        (나머지 파라미터): 추천 함수(recommend_for_context)에 필요한 객체들
        
    Returns:
        dict: 평가 지표 (Recall@K, MRR@K, HitRate@K, EVALUATED)
    """
    print(f"\n--- 6. 랭킹 평가 (K={K}) ---")
    recalls, mrrs, hits = [], [], []
    total = 0
    rec_cache = {} # (최적화) 동일 context에 대한 추천 결과 캐싱

    # 테스트셋의 모든 (유저-아이템) 상호작용(row)을 순회
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="[Eval]"):
        ctx = row.get("context")
        true_code = str(row.get("product_code", "")).zfill(10) # 유저가 실제 입은 정답 아이템
        if not isinstance(ctx, str) or not true_code:
            continue

        # (최적화) 이 context를 처음 보는 경우에만 추천 함수 호출
        if ctx not in rec_cache:
            rec_df = recommend_for_context(ctx, max(K, 50), df_cf, df_dress, model, ctx2codes, bert_matrix, item_index_map)
            rec_cache[ctx] = rec_df["product_code"].tolist()
        
        top_codes = rec_cache[ctx][:K] # 캐시된 결과에서 Top-K 추출

        # '정답 아이템'이 'Top-K 추천 목록'에 포함되는지(hit) 확인
        hit = 1 if true_code in top_codes else 0
        hits.append(hit)
        recalls.append(hit) # (이 시나리오에서는 Recall@K == HitRate@K)

        mrr = 0.0
        if hit:
            # 맞췄다면, 몇 위(rank)로 맞췄는지 확인 (예: 1위 -> 1/1, 3위 -> 1/3)
            for rank, c in enumerate(top_codes, start=1):
                if c == true_code:
                    mrr = 1.0 / rank
                    break
        mrrs.append(mrr)
        total += 1

    avg = lambda x: float(np.mean(x)) if len(x) else 0.0
    return {f"Recall@{K}": avg(recalls), f"MRR@{K}": avg(mrrs), f"HitRate@{K}": avg(hits), "EVALUATED": total}


# -----------------------------
# 8) 메인 파이프라인
# -----------------------------
def main(topk=TOPK_EVAL):
    """메인 실행 함수: 1~7단계를 순차적으로 실행"""
    
    start_time = time.time()
    
    # 1) 로드 & 정리
    df_train, df_test, df_cf, df_art = load_and_prepare()
    print(f"[LOAD] train={len(df_train):,}, test={len(df_test):,}, cf={len(df_cf):,}, articles={len(df_art):,}")

    # 2) BERT 임베딩(CBF)
    df_dress, bert_matrix, item_index_map = build_bert_embeddings(df_train, df_art)
    print(f"[CBF] built: items={bert_matrix.shape[0]:,}, dims={bert_matrix.shape[1]}")

    # 3) 학습 피처 구성 (X, y 생성)
    X, y, feat_train, ctx2codes = build_train_features(df_train, df_cf, df_dress, bert_matrix, item_index_map)
    print(f"[FEAT] X={X.shape}, y={y.shape}")

    # 4) XGBoost 랭킹 모델 학습
    model = train_xgb(X, y)

    # 5) 랭킹 성능 평가
    metrics = evaluate_topk_on_test(df_test, topk, df_cf, df_dress, model, ctx2codes, bert_matrix, item_index_map)
    print(f"\n[RANK] Top-{topk} => {metrics}")

    # 6) 데모: 테스트셋에서 가장 빈도가 높은 컨텍스트로 Top-10 실제 추천
    try:
        demo_ctx = df_test["context"].mode().iat[0]
        demo_top = recommend_for_context(demo_ctx, 10, df_cf, df_dress, model, ctx2codes, bert_matrix, item_index_map)
        print(f"\n[DEMO] Top-10 for context = '{demo_ctx}'\n{demo_top.to_string(index=False)}")
    except Exception as e:
        print(f"[DEMO] skip: {e}")
        
    print(f"\n--- Total Time: {time.time() - start_time:.2f} sec ---")


if __name__ == "__main__":
    main()
