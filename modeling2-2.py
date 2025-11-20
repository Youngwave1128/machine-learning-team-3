# ====================== (두 번째 블록: 추천 + 평가) ======================

import pandas as pd                                  # 재임포트(스크립트 분리 실행을 가정)
import numpy as np
import sys
import traceback

# K-Fold 및 RMSE 계산 모듈
from sklearn.model_selection import KFold            # 교차검증 KFold
from sklearn.metrics import mean_squared_error       # RMSE 계산용 MSE

# --- 0. 설정 ---
# (4단계: 최종 추천 + K-Fold RMSE + Test Set RMSE/Precision/Recall/F1)

# 입력 파일 (앞 단계 출력물)
file_train_clustered = "renttherunway_clustered.csv"  # 학습셋(문맥 포함)
file_test_clustered  = "modcloth_clustered.csv"       # 테스트셋(문맥 포함)
file_cf_model        = "context_item_matrix.csv"      # (context, item) 평균 평점 테이블

# 아이템 이름 매핑용 원본 파일
file_articles = "articles.csv"                        # product_code → prod_name 매핑

# 추천 및 평가 기준
TOP_N = 3                                             # Top-N 추천 개수
RATING_THRESHOLD = 8.0                                # Relevant 정의(10점 만점 기준 8점 이상)
K_FOLDS = 5                                           # K-Fold 횟수

print(f"--- 4단계: 최종 추천 및 모델 평가 시작 ---")                   # 섹션 로그
print(f"(평가 기준: 10점 만점 / Top {TOP_N} 추천 / '좋아함' >= {RATING_THRESHOLD}점)")

try:
    # --- 1. 필수 데이터 로드 ---
    print("\n--- 1: 필수 파일 로드 (모든 product_code를 'str'로 통일) ---")  # 섹션 로그

    # 1a. (3단계 모델) Context-Item 평점 테이블 로드
    df_cf_model = pd.read_csv(file_cf_model, index_col='context')                # 인덱스=문맥
    df_cf_model.columns = df_cf_model.columns.astype(str).str.strip()            # 컬럼 문자열/공백 정리
    print(f"  - CF Score 모델 로드 완료 (10점 만점 기준) (Shape: {df_cf_model.shape})")

    # 1b. (2단계 테스트셋) 평가용 로드
    df_test = pd.read_csv(file_test_clustered, dtype={'product_code': str})      # 테스트셋 로드
    df_test['product_code'] = df_test['product_code'].str.strip()                # 코드 공백 제거

    # [수정] 'quality' → 'rating'으로 이름 변경 및 5→10점 스케일 변환
    if 'quality' in df_test.columns:                                             # 일부 데이터셋 호환 처리
        df_test = df_test.rename(columns={'quality': 'rating'})                  # 컬럼명 통일
        print("  - 테스트 데이터 'quality' -> 'rating' 컬럼명 변경 완료.")
        if 'rating' in df_test.columns:
            df_test['rating'] = df_test['rating'] * 2.0                          # 5점만점 → 10점만점
            print("  - 테스트 데이터 'rating' 5점 -> 10점 스케일링 완료.")

    print(f"  - 테스트 데이터 로드 완료 ({len(df_test):,}건)")

    # 1c. (2단계 훈련셋) K-Fold 평가용 로드
    df_train = pd.read_csv(file_train_clustered, dtype={'product_code': str})    # 학습셋 로드
    df_train['product_code'] = df_train['product_code'].str.strip()              # 코드 공백 제거
    print(f"  - 훈련 데이터 로드 완료 ({len(df_train):,}건)")

    # 1d. 아이템 이름(prod_name) 매핑용 로드
    df_articles_names = pd.read_csv(
        file_articles,
        encoding='latin-1',
        dtype={'product_code': str},
        usecols=['product_code', 'prod_name']                                    # 필요한 컬럼만
    ).drop_duplicates(subset=['product_code'])                                    # 코드 중복 제거
    df_articles_names['product_code'] = df_articles_names['product_code'].str.strip()  # 공백 제거
    print("  - H&M 아이템 이름 정보 로드 완료.")

    print("\n" + "="*50)

    # --- 2. 테스트셋에서 랜덤 타겟 유저 1명 선택 ---
    target_user = df_test.dropna(subset=['body_cluster']).sample(1).iloc[0]      # 군집 있는 유저 중 랜덤 1
    target_context = target_user['context']                                      # 해당 유저의 문맥

    print(f"--- 2: 타겟 유저 (랜덤 1명) 프로필 ---")
    print(f"  - User ID (ModCloth): {target_user['user_id']}")                   # 유저 ID
    print(f"  - 렌트 이유 (Context):  {target_user['rented for']}")              # 착용 용도
    print(f"  - 신체 정보 (H):        {target_user['height_cm']:.0f} cm")        # 키
    print(f"  - 신체 정보 (Size):     {target_user['size_scaled']:.3f} (Scaled)")# 스케일 사이즈
    print(f"  - 신체 정보 (Bra):      {target_user['bra size']}")                # 브라 사이즈
    print(f"  - 신체 정보 (Cup):      {target_user['cup size']}")                # 컵 사이즈
    print(f"  ▶ 예측된 체형 그룹:     C{int(target_user['body_cluster'])}")      # 군집 ID
    print(f"  ▶ 최종 Context (유사 그룹): {target_context}")                      # 문맥 키
    print("="*50)

    # --- 3. 동일 Context의 학습셋 프로필(탐색용) ---
    similar_group_users = df_train[df_train['context'] == target_context]        # 같은 문맥의 학습 유저들
    if not similar_group_users.empty:
        print(f"--- 3: '{target_context}' 그룹 훈련 데이터 프로필 (참고) ---")
        print(f"  - 훈련 세트의 총 유저 수: {len(similar_group_users):,} 명")   # 인원 수
        print(f"  - [평균] 신체 정보 (H): {similar_group_users['height_cm'].mean():.0f} cm")  # 평균 키
    else:
        print(f"--- 3: '{target_context}' 그룹 훈련 데이터 프로필 (참고) ---")
        print("  - 훈련 세트에 이 Context에 해당하는 유저가 없습니다.")
    print("="*50)

    # --- 4. CF Score 기반 최종 추천 ---
    print(f"--- 4: 최종 추천 (Top {TOP_N}) ---")
    print("="*50)

    if target_context in df_cf_model.index:                                       # 문맥이 모델에 존재하는지
        cf_scores = df_cf_model.loc[target_context]                               # 문맥의 아이템 점수 시리즈
        cf_scores_sorted = cf_scores.dropna().sort_values(ascending=False)        # 높은 점수 순 정렬

        if cf_scores_sorted.empty:
            print(f"\n[추천 결과] '{target_context}' 그룹은 훈련 데이터에서 평점을 매긴 기록이 없습니다.")
        else:
            top_n_items = cf_scores_sorted.head(TOP_N)                            # Top-N 아이템 선택
            df_recs = top_n_items.reset_index()                                   # DataFrame으로 변환
            df_recs.columns = ['product_code', 'cf_score (Group Avg Rating / 10.0)']  # 컬럼명 정리(10점 만점)
            df_recs = pd.merge(df_recs, df_articles_names, on='product_code', how='left')  # 이름 매핑
            print(f"\n[추천 결과] '{target_context}' 그룹이 가장 좋아한 아이템 Top {len(df_recs)} (CF Score 기준)")
            print(df_recs.to_string(index=False))                                 # 표 형태 출력
    else:
        print(f"\n[추천 결과] '{target_context}' 그룹은 훈련 데이터(평점 모델)에 존재하지 않습니다.")  # 문맥 미존재

    # --- 5. 성능 평가 1: 훈련 데이터 K-Fold RMSE ---
    print("\n" + "="*50)
    print(f"--- 5: 성능 평가 1: 훈련 데이터 K-Fold (K={K_FOLDS}) RMSE ---")

    if 'rating' not in df_train.columns:                                          # 평점 컬럼 검증
        print("[오류] 훈련 데이터에 'rating' 컬럼이 없어 K-Fold 평가를 건너뜁니다.")
    else:
        df_eval = df_train.dropna(subset=['context', 'product_code', 'rating'])   # 평가 가능 행만

        if len(df_eval) > 100:                                                    # 최소 표본 수 체크(임계값 예시)
            kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)           # KFold 생성
            rmse_scores = []                                                      # 폴드별 RMSE 저장 리스트

            for fold, (train_index, test_index) in enumerate(kf.split(df_eval), 1):  # 각 폴드 반복
                train_fold = df_eval.iloc[train_index]                            # 폴드 학습 분할
                test_fold = df_eval.iloc[test_index]                              # 폴드 검증 분할

                # 1) 폴드 학습 데이터로 CF 모델(평균) 생성
                df_grouped_fold = train_fold.groupby(['context', 'product_code'])['rating'].mean().reset_index()
                cf_model_fold = df_grouped_fold.pivot(index='context', columns='product_code', values='rating')
                cf_model_fold.columns = cf_model_fold.columns.astype(str)        # 컬럼 타입 정리

                # 2) 폴드 검증 데이터로 예측값 수집
                ground_truth = []                                                # 정답 리스트
                predictions = []                                                 # 예측 리스트
                valid_contexts = set(cf_model_fold.index)                        # 유효 문맥 집합
                valid_items = set(cf_model_fold.columns)                         # 유효 아이템 집합

                for _, row in test_fold.iterrows():
                    context, item, truth = row['context'], row['product_code'], row['rating']  # 레코드
                    if context in valid_contexts and item in valid_items:        # 문맥/아이템 모두 존재
                        predicted_rating = cf_model_fold.loc[context, item]      # 예측(그룹 평균 평점)
                        if not pd.isna(predicted_rating):                        # 결측 제외
                            ground_truth.append(truth)                           # 실제값 저장
                            predictions.append(predicted_rating)                 # 예측값 저장

                # 3) 폴드별 RMSE 계산
                if len(predictions) > 0:
                    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))  # RMSE 계산
                    rmse_scores.append(rmse)                                       # 리스트에 추가
                    print(f"  - Fold {fold}/{K_FOLDS} 완료: RMSE = {rmse:.4f}")    # 폴드 결과 로그
                else:
                    print(f"  - Fold {fold}/{K_FOLDS} 완료: 평가 가능한 예측이 없습니다.")  # 매칭 없음

            # 4) 폴드 평균 RMSE 출력
            if len(rmse_scores) > 0:
                mean_rmse = np.mean(rmse_scores)                                 # 평균 RMSE
                print(f"\n[성능 평가 1 결과 (Train K-Fold)]")
                print(f"  ▶ K-Fold (K=5) 평균 RMSE: {mean_rmse:.4f} (10점 만점 기준)")  # 결과 출력
            else:
                print("\n[성능 평가 1 결과 (Train K-Fold)]\n  - 교차 검증 중 RMSE를 계산하지 못했습니다.")
        else:
            print("[오류] 훈련 데이터가 너무 적어 K-Fold 평가를 건너뜁니다.")             # 표본 부족

    print("="*50)

    # --- 6. 성능 평가 2: 테스트 데이터 (RMSE & Precision/Recall/F1) ---
    print(f"--- 6: 성능 평가 2: 테스트 데이터 (RMSE & P/R/F1@{TOP_N}) ---")

    if 'rating' not in df_test.columns:                                          # 테스트 평점 유무 확인
        print("[알림] 테스트 데이터에 'rating'(quality) 컬럼이 없습니다.")
        print("       테스트셋 성능 평가(RMSE, P/R/F1)를 건너뜁니다.")
        print("="*50)
    else:
        test_ground_truth_rmse = []                                              # RMSE 실제값
        test_predictions_rmse = []                                               # RMSE 예측값
        test_precisions, test_recalls, test_f1s = [], [], []                     # 분류 지표 저장
        test_contexts = df_test['context'].dropna().unique()                     # 평가 대상 문맥들

        if len(test_contexts) == 0:                                              # 문맥 없음
            print("[오류] 테스트 데이터에 유효한 Context가 없어 평가를 건너뜁니다.")
            sys.exit()

        for target_context in test_contexts:                                     # 문맥별로 평가
            # 1) 예측(Top-N 추천)
            if target_context in df_cf_model.index:
                cf_scores = df_cf_model.loc[target_context]                      # 해당 문맥 점수
                recommended_items = set(cf_scores.nlargest(TOP_N).index)         # 상위 N개 아이템
            else:
                recommended_items = set()                                        # 문맥 미존재 시 공집합

            # 2) 정답(관련 아이템: 평점 임계 이상)
            test_subset = df_test[df_test['context'] == target_context]          # 문맥 해당 행들
            relevant_items = set(test_subset[test_subset['rating'] >= RATING_THRESHOLD]['product_code'])  # 정답셋

            if len(relevant_items) == 0:                                         # 정답 없음이면 스킵
                continue

            # 3) 교집합 개수(히트)
            hits = len(recommended_items.intersection(relevant_items))           # 추천과 정답 교집합

            # 4) P/R/F1 계산
            precision = hits / TOP_N if TOP_N > 0 else 0.0                       # 정밀도
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0  # 재현율
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # F1

            test_precisions.append(precision)                                    # 누적
            test_recalls.append(recall)
            test_f1s.append(f1)

            # --- (Test RMSE) 예측-실제 매칭 수집 ---
            for _, row in test_subset.iterrows():                                # 문맥 내 각 행 순회
                item, truth = row['product_code'], row['rating']                 # 아이템/실제 평점
                if target_context in df_cf_model.index and item in df_cf_model.columns:
                    predicted_rating = df_cf_model.loc[target_context, item]     # 예측 평점(그룹 평균)
                    if not pd.isna(predicted_rating) and not pd.isna(truth):     # 결측 제외
                        test_ground_truth_rmse.append(truth)                     # 실제값 저장
                        test_predictions_rmse.append(predicted_rating)           # 예측값 저장

        # 6c. (Test RMSE) 최종 결과
        if len(test_predictions_rmse) > 0:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth_rmse, test_predictions_rmse))  # RMSE
            print(f"\n[성능 평가 2 결과 (Test Set)]")
            print(f"  ▶ 테스트 RMSE: {test_rmse:.4f} (총 {len(test_predictions_rmse):,}건 비교) (10점 만점 기준)")
        else:
            print(f"\n[성능 평가 2 결과 (Test Set)]")
            print("  - 테스트 RMSE: 계산 불가 (예측값/실제값 매칭 0건)")

        # 6d. (Test P/R/F1) 최종 결과
        if len(test_precisions) > 0:
            avg_precision = np.mean(test_precisions)                             # 평균 정밀도
            avg_recall = np.mean(test_recalls)                                   # 평균 재현율
            avg_f1 = np.mean(test_f1s)                                           # 평균 F1
            print(f"  ▶ Precision@{TOP_N}: {avg_precision:.4f} (={avg_precision*100:.2f}%)")
            print(f"  ▶ Recall@{TOP_N}:    {avg_recall:.4f} (={avg_recall*100:.2f}%)")
            print(f"  ▶ F1-Score@{TOP_N}:  {avg_f1:.4f}")
            print(f"  (총 {len(test_contexts):,}개 Context 중 {len(test_precisions):,}개 Context 평가)")
        else:
            print(f"  ▶ Precision/Recall/F1: 계산 불가 ({RATING_THRESHOLD}점 이상 평가 0건)")
        print("="*50)

    print("\n--- 모든 작업 완료 ---")                                            # 전체 파이프라인 종료

except FileNotFoundError as e:
    print(f"[파일 로드 오류] {e}.")                                              # 필수 파일 누락
    print("필요한 파일(renttherunway_clustered.csv, modcloth_clustered.csv, context_item_matrix.csv, articles.csv)이 모두 있는지 확인하세요.")
except KeyError as e:
    print(f"[KeyError] {e}.")                                                    # 컬럼명 오류
    print("CSV 파일에 'context', 'product_code', 'body_cluster', 'rating', 'quality' 등의 필수 컬럼명이 존재하는지 확인하세요.")
except ImportError as e:
    print("[오류] 'scikit-learn' 라이브러리가 필요합니다. 'pip install scikit-learn'을 실행해주세요.")  # 의존성
except Exception as e:
    print(f"처리 중 예기치 않은 오류 발생: {e}")                                # 기타 예외
    traceback.print_exc()                                                        # 스택 출력
