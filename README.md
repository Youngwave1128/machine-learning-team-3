# 👗 Hybrid Fashion Recommendation System
CF + CBF + XGBoost 기반 Hybird 드레스 추천 모델

<br>

## Overview
이 프로젝트는 드레스 대여 추천 시스템으로, 사용자의 체형 정보와 대여 목적(rent_for)을 기반으로 최적의 드레스를 추천합니다. 

<br>

핵심 기술:
- **User-based CBF**: 신체 정보 기반 클러스터링 (K-Prototypes)
- **Collaborative Filtering**: 그룹(Context)별 평점(rating) 및 핏(fit) 점수 계산
- **Content-based Filtering**: BERT 임베딩을 이용한 스타일 유사도  
- **XGBoost**: 3가지 점수를 통합하여 최종 추천 점수 예측
<br>

## Architecture

1. **데이터 준비**: 초기 데이터 로드 및 전처리
2. **유저 그룹화 (User-based CBF)**: 사용자를 특성별로 그룹화
3. **CF 피처 계산**: `Rating` 및 `Fit` 점수를 각각 계산
4. **CBF 피처 계산**: BERT를 이용한 스타일 점수 계산
5. **하이브리드 모델 (XGBoost)**: 3단계(Rating, Fit)와 4단계(Style)의 점수들을 개별 피처(feature)로 사용하여 최종 랭킹 ([HYBRID_MODEL.md](HYBRID_MODEL.md))
<br>

## Avtice Learning

* **1. 평점 결측치 보간 (Data Imputation)**
    * 단순히 결측치를 제거하지 않고, `review` 텍스트의 **감성 분석(Sentiment Analysis)** 을 통해 빈 `rating` 값을 예측하여 채워넣었습니다.

* **2. BERT를 활용한 스타일 피처 생성**
    * 아이템의 텍스트(`prod_name`, `detail_desc`)를 **BERT (SentenceTransformer)** 를 이용해 고차원 벡터로 변환했습니다.
    * 이를 통해 사용자의 과거 스타일과 아이템 간의 '스타일 유사도'를 정량적으로 계산할 수 있었습니다.

* **3. 하이브리드 랭킹 모델 (XGBoost)**
    * CF(평점, 핏) 점수와 CBF(스타일) 점수를 **XGBoost 모델의 개별 피처(feature)** 로 사용하여, 데이터에 기반한 최적의 추천 가중치를 학습시켰습니다.

