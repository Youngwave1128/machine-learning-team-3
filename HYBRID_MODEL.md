## Hybrid Model 상세 함수 설명 (XGBoost + BERT)
<br>

### **개요**

이 추천 시스템은 CF(협업 필터링) 평점, CBF(콘텐츠 기반 필터링, BERT 임베딩)을 각각 계산한 뒤
XGBoost 회귀 모델을 통해 **사용자 체형 / 대여 목적별로 드레스 아이템을 Top-N 랭킹 추천**합니다.

<br>

### **1. 데이터 로드 및 전처리 (Code Section 1)**

4개의 CSV 파일을 로드하고, 추천의 기준이 되는 `context`(상황) 정보를 정리합니다.
특히 CF 데이터 내 `_rent_` 오타를 수정해 `context` 키를 통일합니다.

```python
# 컨텍스트 데이터 통합 및 오타 수정
if "context" in df_cf.columns:
    cf_ctx_sample = df_cf["context"].iloc[0]
    if "_rent_" in cf_ctx_sample:
        df_cf["context"] = df_cf["context"].str.replace("_rent_", "_", regex=False)
```

**실행결과**

```
--- 1. 데이터 로드 및 전처리 ---
[정보] Train context 샘플: cluster_5_wedding_black_tie
[정보] CF context 원본: cluster_0_rent_date_general
[수정] CF context에서 '_rent_' 제거 중...
[수정 후] CF context: cluster_0_date_general
[LOAD] train=92,879, test=18,416, cf=39,403, articles=105,542

[디버깅] Context 샘플 (수정 후):
  Train: cluster_5_wedding_black_tie
  CF:    cluster_0_date_general
  Test:  cluster_8_wedding_black_tie
```

<br>

### **2. CBF (BERT 기반 임베딩 구축, Code Section 2)**

CBF(Content-Based Filtering)에서 사용할 **스타일 임베딩(Embedding Matrix)** 을 생성합니다.
 
- 해당 CBF는 드레스의 이름·상세설명 텍스트를 BERT로 임베딩해 아이템 프로필을 만들고, 
 특정 context(체형 클러스터+rent_for)의 과거 착용 아이템 임베딩을 평균해 사용자 프로필을 구성한 뒤 코사인 유사도로 유사도를 산출

1. `articles.csv`에서 `prod_name` + `detail_desc`를 합쳐 `style_text`를 구성
2. `SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')` 모델 로드
3. 전체 드레스의 `style_text`를 384차원 벡터로 변환하여 `bert_matrix` 완성

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
bert_matrix = model.encode(df_dress['style_text'].tolist(), show_progress_bar=True)
```

**실행결과**

```
--- 2. CBF: BERT 스타일 모델 구축 ---
  - 훈련 데이터의 고유 아이템 수: 3,766
  - Articles에서 매칭된 아이템: 6,994
  - Dress 타입 아이템: 6,958
BERT 모델 로드 완료.
Batches: 100%
 118/118 [00:05<00:00, 36.59it/s]
[CBF] BERT built: items=3,766, dims=384
```

<br>

### **3. `compute_cbf_for_context(ctx)` 함수 정의 (Code Section 3)**

특정 `context`(예: `cluster_2_party`)의 평균 스타일 벡터를 계산하고
전체 드레스와의 **코사인 유사도(Cosine Similarity)** 를 통해 `cbf_style_score`를 생성합니다.

```python
def compute_cbf_for_context(ctx):
    """
    CBF 전용 점수: 아이템 '콘텐츠(텍스트 임베딩)'만으로 컨텍스트-아이템 유사도 계산
    """
    if ctx in _cbf_cache:
        return _cbf_cache[ctx]
    
    codes = ctx2codes.get(ctx, [])
    idxs = [item_index_map.get(c) for c in codes if c in item_index_map]

    if not idxs:
        cbf = pd.DataFrame({
            "product_code": df_dress["product_code"].values,
            "cbf_style_score": np.zeros(len(df_dress))
        })
    else:
        style_vec = bert_matrix[idxs].mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(style_vec, bert_matrix).ravel()
        cbf = pd.DataFrame({
            "product_code": df_dress["product_code"].values,
            "cbf_style_score": scores
        })
    
    _cbf_cache[ctx] = cbf
    return cbf
```

**실행결과**

```
cf_rating_score mean=9.0554
cf_fit_score mean=0.4340
cbf_style_score mean=0.8042
```

<br>

### **4. XGBoost 훈련 및 피처 결합 (Code Section 4 & 5)**

단순 가중합 대신 지도학습으로 최적 피처 조합을 학습하여 최종 점수(평점·선호)를 추정한다는 점에서 XGBoost 메타 모델을 사용합니.

- XGBoost: 비선형 상호작용, 강한 정규화, 빠른 학습과 조기종료로 일반화 성능이 좋아 추천 랭킹·회귀에 자주 쓰이는 앙상블 모델

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    # 메타 회귀: CF/CBF 피처의 최적 결합 가중을 '데이터로부터' 학습  
    objective="reg:squarederror",
    learning_rate=0.1,
    max_depth=5,
    n_estimators=1000,
    early_stopping_rounds=50
)

xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)
```

**실행결과**

```
--- 4. XGBoost 모델 학습 ---
[0]	validation_0-rmse:1.37918
[50]	validation_0-rmse:1.09139
[92]	validation_0-rmse:1.09826

[METRIC] RMSE=1.0910 | R²=0.4256

[XGB] Feature Importance:
        feature  importance
cf_rating_score    0.839935
        n_users    0.121508
   cf_fit_score    0.030348
cbf_style_score    0.008209
```

<br>

### **5. `recommend_for_context(ctx)` 함수 정의 (Code Section 6)**

특정 `context`에 대해 CF, CBF 점수를 병합하고
학습된 XGBoost 모델로 최종 추천 점수를 예측합니다.

```python
def recommend_for_context(ctx, top_n=10):
    base = df_cf[df_cf["context"] == ctx][["product_code","cf_rating_score","cf_fit_score","n_users"]].copy()
    if base.empty:
        return pd.DataFrame(columns=["product_code","prod_name","final_score"])
    
    cbf_ctx = compute_cbf_for_context(ctx)
    base = base.merge(cbf_ctx, on="product_code", how="left").fillna({"cbf_style_score":0.0})
    X_all = base[feature_cols].values

    base["final_score"] = xgb_model.predict(X_all)
    base = base.merge(df_dress[["product_code","prod_name"]], on="product_code", how="left")
    return (
        base.sort_values("final_score", ascending=False)
            .head(top_n)[["product_code","prod_name","final_score"]]
            .reset_index(drop=True)
    )
```

**실행결과**

```
--- 6. 랭킹 평가 ---

[DEMO] Top-10 for context = cluster_3_wedding_black_tie
product_code           prod_name  final_score
  0000547857                Knut     9.986682
  0000653108        Garden Dress     9.986682
  0000574684     Dolled up dress     9.986682
  0000784587         Paris Dress     9.986682
  0000824192                Anya     9.986682
  0000716348           Marcie(1)     9.984791
  0000658242 ED Santa Baby dress     9.981354
  0000886099 Olivia Kaftan Dress     9.979516
  0000859788     Femme dress (J)     9.978331
  0000855788      LOGG Rio tunic     9.974014
```

<br>

### **6. `evaluate_topk_on_test(df_test, K=10)` 함수 정의 (Code Section 7)**

테스트 데이터셋에서 Top-K 추천 품질을 평가합니다.
Recall@K, MRR@K, HitRate@K 지표를 계산합니다.

```python
def evaluate_topk_on_test(df_test, K=10):
    recalls, mrrs, hits = [], [], []
    total, rec_cache = 0, {}

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        ctx = row.get("context")
        true_code = str(row.get("product_code", "")).zfill(10)
        if not isinstance(ctx, str) or not true_code:
            continue

        if ctx not in rec_cache:
            rec_df = recommend_for_context(ctx, top_n=max(K, 50))
            rec_cache[ctx] = rec_df["product_code"].tolist()
        top_codes = rec_cache[ctx][:K]

        hit = 1 if true_code in top_codes else 0
        hits.append(hit)
        recalls.append(hit)

        mrr = 0.0
        if hit:
            for rank, c in enumerate(top_codes, start=1):
                if c == true_code:
                    mrr = 1.0 / rank
                    break
        mrrs.append(mrr)
        total += 1

    avg = lambda x: float(np.mean(x)) if len(x) else 0.0
    return {
        f"Recall@{K}": avg(recalls),
        f"MRR@{K}": avg(mrrs),
        f"HitRate@{K}": avg(hits),
        "EVALUATED": total
    }
```

**실행결과**

```
[RANKING] Top-10 metrics
  Recall@10: 0.0131
  MRR@10: 0.0041
  HitRate@10: 0.0131
  EVALUATED: 18416
```

<br>


