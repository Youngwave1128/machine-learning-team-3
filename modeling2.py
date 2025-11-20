# -*- coding: utf-8 -*-
"""
(2단계) 혼합형 군집화(K-Prototypes)로 body_cluster와 context 생성
(3단계) (context, product_code) 평균 평점 pivot → CF 점수 테이블 저장

입력:  renttherunway_processed_final.csv, modcloth_processed_final.csv
출력:  renttherunway_clustered.csv, modcloth_clustered.csv, context_item_matrix.csv
"""

import sys, traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from kmodes.kprototypes import KPrototypes

# --------------------
# 설정
# --------------------
K_CLUSTERS = 10
IN_TRAIN   = "renttherunway_processed_final.csv"
IN_TEST    = "modcloth_processed_final.csv"

OUT_TRAIN  = "renttherunway_clustered.csv"
OUT_TEST   = "modcloth_clustered.csv"
OUT_CF     = "context_item_matrix.csv"

# --------------------
# 유틸: 컬럼 자동 매핑
# --------------------
def map_first_existing(df: pd.DataFrame, candidates, final_name):
    df.columns = df.columns.str.strip()
    for c in candidates:
        if c in df.columns:
            if c != final_name:
                df.rename(columns={c: final_name}, inplace=True)
            print(f"[MAP] {final_name} ← {c}")
            return
    raise KeyError(f"필수 컬럼 '{final_name}' 없음. 후보={candidates}\n현재={list(df.columns)}")

def harmonize_schema(df: pd.DataFrame, is_test: bool=False) -> pd.DataFrame:
    # 현재 데이터 기준: size_scaled_0_to_10 존재 → size_scaled로 통일
    map_first_existing(df, ['size_scaled', 'size_scaled_0_to_10'], 'size_scaled')
    map_first_existing(df, ['bra size','bra_size','bra-size'], 'bra size')
    map_first_existing(df, ['cup size','cup_size','cup-size'], 'cup size')
    map_first_existing(df, ['rented for','rented_for','rented-for'], 'rented for')
    map_first_existing(df, ['product_code','product id','product_id','article_id'], 'product_code')
    map_first_existing(df, ['rating','quality'], 'rating')  # (quality였다면 나중 단계에서 ×2 가능)
    return df

# --------------------
# 파이프라인 시작
# --------------------
try:
    print(f"--- (2단계+3단계) 시작 (K={K_CLUSTERS}) ---")

    # 1) 훈련셋 로드 & 스키마 통일
    df_train = pd.read_csv(IN_TRAIN)
    df_train = harmonize_schema(df_train, is_test=False)

    # 클러스터링 피처 정의(통일된 이름 기준)
    CLUSTER_FEATURES     = ['height_cm', 'size_scaled', 'bra size', 'cup size']
    NUMERICAL_FEATURES   = ['height_cm', 'size_scaled']
    CATEGORICAL_FEATURES = ['bra size', 'cup size']

    # 결측/타입 보정
    df_tr_clu = df_train[CLUSTER_FEATURES].copy()
    for col in NUMERICAL_FEATURES:
        df_tr_clu[col] = pd.to_numeric(df_tr_clu[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        df_tr_clu[col] = df_tr_clu[col].astype(str)

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_tr_clu[NUMERICAL_FEATURES]   = num_imputer.fit_transform(df_tr_clu[NUMERICAL_FEATURES])
    df_tr_clu[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(df_tr_clu[CATEGORICAL_FEATURES])

    scaler = StandardScaler()
    df_tr_clu[NUMERICAL_FEATURES] = scaler.fit_transform(df_tr_clu[NUMERICAL_FEATURES])

    train_matrix = df_tr_clu.values
    cat_idx = [df_tr_clu.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    # 2) K-Prototypes 학습
    print("[INFO] K-Prototypes 학습 중 ...")
    kproto = KPrototypes(n_clusters=K_CLUSTERS, init='Huang', n_init=10,
                         random_state=42, verbose=1, n_jobs=-1)
    clusters_train = kproto.fit_predict(train_matrix, categorical=cat_idx)
    print("[INFO] K-Prototypes 완료.")

    # 3) 훈련셋에 body_cluster/context 부여 및 저장
    df_train.loc[df_tr_clu.index, 'body_cluster'] = clusters_train
    df_train['body_cluster'] = df_train['body_cluster'].astype(float).astype('Int64')
    df_train['context'] = 'C' + df_train['body_cluster'].astype(str) + '_' + df_train['rented for']
    df_train.to_csv(OUT_TRAIN, index=False, encoding='utf-8-sig')
    print(f"[SAVE] {OUT_TRAIN}")

    # 4) 테스트셋 로드 & 스키마 통일
    df_test = pd.read_csv(IN_TEST)
    df_test = harmonize_schema(df_test, is_test=True)

    # (선택) 테스트가 5점 척도였다면 10점으로 맞추고 싶을 때:
    # if 'quality'가 원본이었다면 위에서 rating으로 rename됨 → 필요 시 스케일 조정
    # 예: df_test['rating'] = df_test['rating'] * 2.0

    # 결측/타입 보정(훈련에서 fit된 imputer/scaler 사용)
    df_te_clu = df_test[CLUSTER_FEATURES].copy()
    for col in NUMERICAL_FEATURES:
        df_te_clu[col] = pd.to_numeric(df_te_clu[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        df_te_clu[col] = df_te_clu[col].astype(str)

    df_te_clu[NUMERICAL_FEATURES]   = num_imputer.transform(df_te_clu[NUMERICAL_FEATURES])
    df_te_clu[CATEGORICAL_FEATURES] = cat_imputer.transform(df_te_clu[CATEGORICAL_FEATURES])
    df_te_clu[NUMERICAL_FEATURES]   = scaler.transform(df_te_clu[NUMERICAL_FEATURES])

    test_matrix = df_te_clu.values
    clusters_test = kproto.predict(test_matrix, categorical=cat_idx)

    df_test.loc[df_te_clu.index, 'body_cluster'] = clusters_test
    df_test['body_cluster'] = df_test['body_cluster'].astype(float).astype('Int64')
    df_test['context'] = 'C' + df_test['body_cluster'].astype(str) + '_' + df_test['rented for']
    df_test.to_csv(OUT_TEST, index=False, encoding='utf-8-sig')
    print(f"[SAVE] {OUT_TEST}")

    # 5) (3단계) CF 점수 테이블 생성 (context, product_code 평균 평점)
    df_grouped = df_train.groupby(['context','product_code'])['rating'].mean().reset_index()
    cf_pivot = df_grouped.pivot(index='context', columns='product_code', values='rating')
    cf_pivot.to_csv(OUT_CF, encoding='utf-8-sig')
    print(f"[SAVE] {OUT_CF}  (shape={cf_pivot.shape})")

    print("\n✅ 완료")

except Exception as e:
    print(f"[ERROR] {e}")
    traceback.print_exc()
    sys.exit(1)



