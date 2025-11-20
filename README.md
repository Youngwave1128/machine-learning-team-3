# 👗 AI-Powered Hybrid Fashion Recommendation System

> **가천대학교 인공지능전공 머신러닝 텀프로젝트**
>
> **작성자:** 조영현 (Department of AI, Gachon Univ.)

## 📖 프로젝트 소개
이 프로젝트는 사용자의 신체 정보, 대여 목적(Context), 그리고 아이템의 스타일 정보를 결합하여 개인화된 의류를 추천하는 **하이브리드 추천 시스템**입니다.

기존 협업 필터링(CF)의 한계인 콜드 스타트 문제를 해결하고, 사용자의 구체적인 상황(예: 결혼식 하객룩, 데이트룩)과 체형에 딱 맞는 옷을 추천하기 위해 **K-Prototypes 군집화, BERT 임베딩, XGBoost** 등 다양한 머신러닝 기법을 앙상블했습니다.

## 🚀 핵심 기능 및 알고리즘
1.  **결측 평점 보완 (Sentiment Analysis):** `RoBERTa` 모델을 활용해 리뷰 텍스트의 감성을 분석하고, 누락된 평점(Rating)을 역산출하여 데이터 밀도 향상.
2.  **사용자 그룹화 (Clustering):** `K-Prototypes`를 사용하여 신체 치수(수치형)와 컵 사이즈(범주형)를 기반으로 유사한 체형과 취향을 가진 사용자 군집(Body Cluster) 생성.
3.  **컨텍스트 기반 CF (Context-Aware CF):** '체형 군집' + '대여 목적(Rented for)'을 결합한 Context를 정의하고, 베이지안 스무딩(Bayesian Smoothing)을 적용한 신뢰도 기반 평점 산출.
4.  **스타일 분석 (Content-Based Filtering):** `Sentence-BERT`를 활용하여 의류의 상품명과 상세 설명을 벡터화하고, 사용자가 선호하는 스타일과의 코사인 유사도 계산.
5.  **하이브리드 랭킹 (Hybrid Ranking):** CF 점수(평점, 핏)와 CBF 점수(스타일)를 피처로 결합하고, `XGBoost` 회귀 모델을 학습시켜 최종 추천 점수 예측.

## 📂 데이터셋
* **RentTheRunway & ModCloth:** 사용자 신체 정보, 평점, 리뷰 데이터.
* **H&M Articles:** 상품 메타데이터 및 상세 설명 (Content-Based Filtering용).

## 🛠️ 기술 스택 (Tech Stack)
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy, NLTK
* **Machine Learning:** Scikit-learn, KModes (K-Prototypes), XGBoost
* **Deep Learning / NLP:** PyTorch, HuggingFace Transformers (RoBERTa), Sentence-Transformers (BERT)
* **Visualization:** Matplotlib, Seaborn

## 📂 파일 구조 설명
| 파일명 | 설명 |
|:--- |:--- |
| `Pro.py` | 리뷰 데이터의 N-gram 분석을 통한 키워드 탐색 및 EDA 수행 |
| `Process.py` | H&M 상품 설명을 TF-IDF로 분석하여 11개 카테고리(TPO) 라벨링 |
| `rating.py` | RoBERTa 감성 분석 모델로 결측된 평점(Rating) 예측 및 보완 |
| `Processing.py` | 데이터 정제, 단위 통일(cm/kg), Train/Test 셋 구성 및 스키마 통일 |
| `modeling2.py` | K-Prototypes 군집화 수행 및 Context-Item 매트릭스(CF Score) 생성 |
| `modeling2-2.py` | CF 모델 기반 추천 및 K-Fold 교차 검증, 기본 성능 평가 수행 |
| **`ML_TermProject.ipynb`** | **[메인 파일]** 전체 파이프라인 통합 실행, BERT 임베딩, XGBoost 하이브리드 모델 학습 및 최종 비교 평가 |

## 📊 성능 평가 결과
| Metric | XGBoost Model | Hybrid (Weighted Sum) | Improvement |
|:---:|:---:|:---:|:---:|
| **Recall@10** | 0.0135 | 0.0127 | **+6.87%** (XGB Wins) |
| **MRR@10** | 0.0040 | 0.0038 | **+5.77%** (XGB Wins) |

> *단순 가중합 방식(Hybrid)보다 XGBoost를 메타 러너로 사용했을 때 추천 성능이 유의미하게 향상됨을 확인했습니다.*

## 🔧 실행 방법 (How to Run)

1. **필수 라이브러리 설치**
   ```bash
   pip install -r requirements.txt