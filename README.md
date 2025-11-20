# 👗 AI-Powered Hybrid Fashion Recommendation System

> **가천대학교 인공지능전공 머신러닝 텀프로젝트**
>
> **작성자:** 조영현 (Department of AI, Gachon Univ.)

## 📖 프로젝트 소개 (Project Overview)
이 프로젝트는 사용자의 **신체 정보**, **대여 목적(TPO)**, 그리고 **아이템의 스타일**을 결합하여 개인화된 의류를 추천하는 **하이브리드 추천 시스템**입니다.

단순한 협업 필터링(CF)의 한계인 **콜드 스타트(Cold Start)** 문제를 해결하고, 사용자의 구체적인 상황(예: 결혼식, 파티, 휴가)에 맞는 옷을 추천하기 위해 **K-Prototypes 군집화**, **BERT 임베딩**, **XGBoost 랭킹** 등 다양한 머신러닝/딥러닝 기법을 앙상블했습니다.

## 🚀 핵심 기술 및 알고리즘 (Key Methodologies)
1.  **결측 평점 보완 (Sentiment Analysis)**
    * `RoBERTa` 모델을 활용해 리뷰 텍스트의 감성을 분석하고, 누락된 평점(Rating)을 역산출하여 데이터 희소성(Sparsity) 문제 완화.
2.  **사용자 그룹화 (User Clustering)**
    * `K-Prototypes` 알고리즘을 사용하여 수치형(키, 몸무게)과 범주형(컵 사이즈) 데이터를 동시에 처리, 유사한 체형과 취향을 가진 **Body Cluster** 생성.
3.  **컨텍스트 기반 CF (Context-Aware CF)**
    * '체형 군집' + '대여 목적(Rented for)'을 결합한 **Context**를 정의하고, 베이지안 스무딩(Bayesian Smoothing)을 적용해 신뢰도 높은 그룹 평점 산출.
4.  **스타일 분석 (Content-Based Filtering)**
    * `Sentence-BERT` (paraphrase-multilingual-MiniLM-L12-v2)를 활용하여 의류의 상품명과 상세 설명을 벡터화하고, 사용자가 선호하는 스타일과의 코사인 유사도 계산.
5.  **하이브리드 랭킹 (Hybrid Ranking with XGBoost)**
    * CF 점수(Rating, Fit)와 CBF 점수(Style), 인기도 등을 피처로 결합하고, **XGBoost** 회귀 모델을 학습시켜 최종 추천 순위 예측.

## 📂 데이터셋 (Datasets)
* **RentTheRunway & ModCloth:** 사용자 신체 정보, 평점, 리뷰 데이터.
* **H&M Articles:** 상품 메타데이터 및 상세 설명 (Content-Based Filtering용).

## 🛠️ 기술 스택 (Tech Stack)
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy, NLTK
* **Machine Learning:** Scikit-learn, KModes (K-Prototypes), XGBoost
* **Deep Learning / NLP:** PyTorch, HuggingFace Transformers (RoBERTa), Sentence-Transformers (BERT)
* **Visualization:** Matplotlib, Seaborn

## 📂 파일 구조 (File Structure)
| 파일명 | 역할 및 설명 |
|:--- |:--- |
| **`ML_TermProject.ipynb`** | **[메인 실행 파일]** 전체 파이프라인 통합 실행, BERT 임베딩, XGBoost 학습 및 최종 시각화 |
| `Pro.py` | 리뷰 데이터 N-gram 분석 및 EDA (탐색적 데이터 분석) |
| `Process.py` | H&M 상품 설명을 TF-IDF로 분석하여 11개 TPO 카테고리 라벨링 |
| `rating.py` | RoBERTa 감성 분석 모델로 리뷰 텍스트 기반 결측 평점 예측 |
| `Processing.py` | 데이터 정제, 단위 통일(cm/kg), Train/Test 분할 및 스키마 통일 |
| `modeling2.py` | K-Prototypes 군집화 수행 및 Context-Item 매트릭스 생성 |
| `modeling2-2.py` | CF 모델 기반 추천, K-Fold 교차 검증 및 기본 성능 평가 |

## 📊 성능 평가 (Results)
| Model | Recall@10 | MRR@10 | HitRate@10 | 비고 |
|:---:|:---:|:---:|:---:|:---:|
| **Hybrid (Weighted Sum)** | 0.0127 | 0.0038 | 0.0127 | 단순 가중합 |
| **XGBoost Hybrid** | **0.0135** | **0.0040** | **0.0135** | **최종 모델** |

> **결론:** 단순 가중합 방식보다 XGBoost를 사용하여 피처 중요도를 학습했을 때, **Recall@10 기준 약 6.87%의 성능 향상**을 확인했습니다.

---

## 🔧 실행 방법 (How to Run)

### 1. 환경 설정
```bash
pip install -r requirements.txt