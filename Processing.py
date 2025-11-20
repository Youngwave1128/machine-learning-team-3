import pandas as pd
import numpy as np
import sys
import traceback
import re  # 정규표현식(Regex) 모듈

# --- [신규] 0. 보조 함수 정의 ---

def convert_height_to_cm(s):
    """
    "5' 8"" 또는 "5ft 6in" 같은 문자열을 cm로 변환합니다.
    """
    if pd.isna(s):
        return np.nan
    
    # 1. 공통 클리닝
    s = str(s).strip().replace('"', '').replace('in', '')
    
    feet = 0
    inches = 0
    
    try:
        if "'" in s: # renttherunway 포맷 ("5' 8")
            parts = s.split("'")
            feet = int(parts[0].strip())
            if len(parts) > 1 and parts[1].strip():
                inches = int(parts[1].strip())
        
        elif "ft" in s: # modcloth 포맷 ("5ft 6")
            parts = s.split("ft")
            feet = int(parts[0].strip())
            if len(parts) > 1 and parts[1].strip():
                inches = int(parts[1].strip())
        
        else: # 인식 불가능한 포맷
            return np.nan
            
        # cm로 변환: ( (피트 * 12) + 인치 ) * 2.54
        total_inches = (feet * 12) + inches
        return total_inches * 2.54
    
    except ValueError: # "apple" 등 숫자로 변환 불가 시
        return np.nan

def convert_weight_to_kg(s):
    """
    "130lbs" 같은 문자열을 kg으로 변환합니다.
    """
    if pd.isna(s):
        return np.nan
    
    # "lbs" 문자 제거 및 공백 제거
    s_cleaned = str(s).lower().replace('lbs', '').strip()
    
    try:
        weight_lbs = float(s_cleaned)
        # 1 lbs = 0.453592 kg
        return weight_lbs * 0.453592
    except ValueError: # "apple" 등 숫자로 변환 불가 시
        return np.nan

def split_bust_size(s):
    """
    "34d", "34d+", "34dd/e" 같은 문자열을 ("34", "d") 또는 ("34", "dd/e")로 분리합니다.
    """
    if pd.isna(s):
        return pd.Series({"bra size": "unknown", "cup size": "unknown"})
    
    s = str(s).strip()
    # 정규식: (숫자들) (문자/슬래시/플러스 등) 로 매칭
    match = re.match(r"^(\d+)([a-z/\\+]+.*)$", s, re.IGNORECASE) 
    
    if match:
        bra_size = match.group(1) # "34"
        # cup size 정제 (소문자화, + 제거, 공백 제거)
        cup_size = match.group(2).lower().replace('+', '').strip() # "d" 또는 "dd/e"
        return pd.Series({"bra size": bra_size, "cup size": cup_size})
    else:
        # "34"만 있거나, "d"만 있거나, 포맷이 안 맞는 경우
        return pd.Series({"bra size": "unknown", "cup size": "unknown"})

def convert_size_to_numeric(s):
    """
    'size' 컬럼의 숫자/문자 혼용 데이터를 숫자로 변환합니다.
    """
    if pd.isna(s):
        return np.nan
    
    s_clean = str(s).strip().lower()

    # 숫자 변환 시도 (e.g., "51", "10", "4")
    try:
        # 정규식을 사용하여 문자열에서 숫자만 추출 (e.g., "size 8" -> 8)
        match = re.search(r"^([\d\.]+)", s_clean)
        if match:
            return float(match.group(1))
        else:
            return np.nan # "apple" 등
    except ValueError:
        return np.nan

# [신규] Min-Max 스케일러(기준) 변수 (Train에서 채워짐)
min_size = 0
max_size = 0


# --- 1. 설정 및 파일 경로 ---

# NumPy 랜덤 시드 고정 (결과 재현을 위해)
np.random.seed(42)

# 입력 파일
file_renttherunway = "renttherunway_final_data.csv"
file_modcloth = "modcloth_final_data.csv"
file_articles = "articles.csv"
file_hm_labeled = "hm_articles_rentfor_top3_11cats.csv"
file_ratings_filled = "renttherunway_filled.csv" # [S1 기능] 결측치 채워진 평점 파일

# 최종 출력 파일
out_train_file = "renttherunway_processed_final.csv"
out_test_file = "modcloth_processed_final.csv"

print("--- [최종 로직 + 단위 통일 + Rating(S1,S2) + Size(S2) + RentFor(S1)] 데이터 처리 시작 ---")

try:
    # --- 2. H&M 드레스 DB 준비 (100% 마스터 목록) ---
    print("--- 1. H&M 드레스 DB 준비 ---")
    
    # 2a. articles.csv에서 'Dress'의 product_code와 prod_name 매핑 정보 추출
    df_articles = pd.read_csv(file_articles, encoding='latin-1', dtype={'product_code': str})
    df_articles['product_code'] = df_articles['product_code'].str.zfill(10) # 0 채우기
    
    df_dress_data = df_articles[df_articles['product_type_name'] == 'Dress'][
        ['product_code', 'prod_name']
    ].drop_duplicates(subset=['product_code'], keep='first')
    
    if df_dress_data.empty:
        print("[오류] 'articles.csv'에서 'Dress' 타입을 찾을 수 없습니다.")
        sys.exit()
        
    print(f"'articles.csv'에서 'Dress' 타입 고유 product_code {len(df_dress_data):,}개와 prod_name 매핑 확인.")
    
    # 2b. 라벨링된 H&M 파일 로드 (유사도 점수(s) 포함)
    df_hm_labeled = pd.read_csv(file_hm_labeled, dtype={'product_code': str})
    df_hm_labeled['product_code'] = df_hm_labeled['product_code'].str.zfill(10) # 0 채우기
    df_hm_labeled_unique = df_hm_labeled.drop_duplicates(subset=['product_code'], keep='first')

    # 2c. H&M 마스터 DB 생성 (이름 + 라벨 정보)
    hm_dresses_db_100pct = pd.merge(
        df_dress_data,
        df_hm_labeled_unique,
        on='product_code',
        how='inner'
    )
    
    if hm_dresses_db_100pct.empty:
        print(f"[오류] 'Dress' 타입의 이름과 라벨링 정보가 일치하는 H&M 데이터가 없습니다.")
        sys.exit()
    
    # [S1 기능] 2d. 'rent_for' 유사도가 모두 0인 product_code 제거
    score_cols = ['rf11_s1', 'rf11_s2', 'rf11_s3']
    hm_dresses_db_100pct[score_cols] = hm_dresses_db_100pct[score_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    mask_has_score = hm_dresses_db_100pct[score_cols].sum(axis=1) > 0
    hm_dresses_db_100pct = hm_dresses_db_100pct[mask_has_score]

    print(f"H&M 마스터 DB (유사도 0점 드레스 제거 후): {len(hm_dresses_db_100pct):,}개")
    if hm_dresses_db_100pct.empty:
        print("[오류] 유효한 'rent_for' 유사도 점수를 가진 H&M 드레스가 없습니다.")
        sys.exit()
    
    # 2e. DB를 효율적인 조회(lookup)를 위해 product_code로 인덱싱
    hm_dresses_db_indexed = hm_dresses_db_100pct.set_index('product_code')
    
    # 2f. 100% product_code 목록
    hm_codes_100pct = hm_dresses_db_100pct['product_code'].values


    # --- 3. 훈련 데이터 (Train Set) 처리 (RentTheRunway) ---
    print(f"\n--- 2. 훈련 데이터 ({file_renttherunway}) 처리 ---")
    
    # 3a. 훈련 데이터 원본 로드
    df_train_raw = pd.read_csv(file_renttherunway)
    print(f"원본 {len(df_train_raw):,}건 로드. 'rating' 결측치 (처리 전): {df_train_raw['rating'].isnull().sum():,}")

    # [S1 기능] 3b. 'rating' 값 업데이트 (renttherunway_filled.csv 사용)
    try:
        df_filled = pd.read_csv(file_ratings_filled)
        df_filled_ratings = df_filled[['user_id', 'item_id', 'rating']].rename(columns={'rating': 'filled_rating'})
        df_train_raw = pd.merge(
            df_train_raw, df_filled_ratings, on=['user_id', 'item_id'], how='left'
        )
        mask = df_train_raw['filled_rating'].notna()
        df_train_raw.loc[mask, 'rating'] = df_train_raw.loc[mask, 'filled_rating']
        df_train_raw = df_train_raw.drop(columns=['filled_rating'])
        print(f"  Train: 'renttherunway_filled.csv' 기준으로 'rating' 업데이트 완료.")
        print(f"  Train: 'rating' 결측치 (처리 후): {df_train_raw['rating'].isnull().sum():,}")
    except FileNotFoundError:
        print(f"[경고] '{file_ratings_filled}' 파일을 찾을 수 없습니다. Rating 채우기를 건너뜁니다.")
    except Exception as e:
        print(f"[경고] Rating 채우기 중 문제 발생: {e}. Rating 채우기를 건너뜁니다.")

    
    # 3c. 'category' == 'dress' 필터링
    df_train = df_train_raw[df_train_raw['category'] == 'dress'].copy()
    print(f"'category'가 'dress'인 훈련 데이터 {len(df_train):,}건 필터링 완료.")

    # 3d. 훈련 데이터 단위 통일
    print("Train: 단위 통일 시작...")
    
    # [S2 기능] 3d-1. Rating (0-10 스케일 유지)
    if 'rating' in df_train.columns:
        df_train['rating'] = pd.to_numeric(df_train['rating'], errors='coerce')
        print("  Train: 'rating' (0-10) 스케일 유지 (float으로 변환).")
    else: 
        print("  Train: 'rating' 컬럼을 찾을 수 없어 표준화를 건너뜁니다.")

    # 3d-2. Height (cm로 변환, 결측치 제거)
    df_train['height'] = df_train['height'].apply(convert_height_to_cm)
    df_train = df_train.dropna(subset=['height']) 
    df_train = df_train.rename(columns={'height': 'height_cm'}) 
    print(f"  Train: Height -> cm 변환 및 결측치 제거. (남은 {len(df_train):,}건)")

    # 3d-3. Weight (lbs -> kg)
    if 'weight' in df_train.columns:
        df_train['weight'] = df_train['weight'].apply(convert_weight_to_kg)
        df_train = df_train.rename(columns={'weight': 'weight_kg'})
        print("  Train: Weight -> kg 변환 완료.")
    else: 
        print("  Train: 'weight' 컬럼을 찾을 수 없어 표준화를 건너뜁니다.")

    # 3d-4. Bust Size (분리, 정제된 함수 사용)
    bust_split_train = df_train['bust size'].apply(split_bust_size)
    df_train[bust_split_train.columns] = bust_split_train
    df_train = df_train.drop(columns=['bust size'])
    print("  Train: 'bust size' -> 'bra size', 'cup size'로 분리 완료.")
    
    # [S2 기능] 3d-5. Clothing Size (0-10 Min-Max 스케일링)
    if 'size' in df_train.columns:
        df_train['size_numeric'] = df_train['size'].apply(convert_size_to_numeric)
        df_train = df_train.dropna(subset=['size_numeric'])
        
        min_size = df_train['size_numeric'].min()
        max_size = df_train['size_numeric'].max()
        print(f"  Train: Size 스케일러 기준값 설정 (Min={min_size}, Max={max_size}).")

        if (max_size - min_size) == 0:
            df_train['size_scaled_0_to_10'] = 5.0 
        else:
            df_train['size_scaled_0_to_10'] = 10 * (df_train['size_numeric'] - min_size) / (max_size - min_size)
            
        df_train = df_train.drop(columns=['size', 'size_numeric'])
        print(f"  Train: 'size' -> 'size_scaled_0_to_10' (0-10점) 정규화 완료. (남은 {len(df_train):,}건)")
    else:
        print("  Train: 'size' 컬럼을 찾을 수 없어 스케일링을 건너뜁니다.")
        sys.exit("오류: Train 데이터에 'size' 컬럼이 없어 스케일링 기준을 설정할 수 없습니다.")
    
    # 3e. item_id -> product_code 컬럼명 변경
    df_train = df_train.rename(columns={'item_id': 'product_code'})
    
    # 3f. product_code 값을 100% H&M DB에서 랜덤 할당 (교체)
    assigned_codes_train = np.random.choice(hm_codes_100pct, size=len(df_train))
    df_train['product_code'] = assigned_codes_train
    print(f"{len(df_train):,}개 레코드에 100% H&M product_code 랜덤 할당 완료.")
    
    # 3g. 할당된 product_code에 매핑되는 *모든 정보* 조회
    assigned_data_train = hm_dresses_db_indexed.loc[assigned_codes_train]

    # 3h. 'prod_name' 컬럼 추가
    df_train['prod_name'] = assigned_data_train['prod_name'].values
    print("'prod_name' 컬럼 추가 완료.")

    # [S1 기능] 3i. 'rented for' 값을 (유사도 > 0 인) Top-3 라벨 중 하나로 교체
    rf_options_train = assigned_data_train[['rf11_top1', 'rf11_top2', 'rf11_top3']].to_numpy()
    rf_scores_train = assigned_data_train[['rf11_s1', 'rf11_s2', 'rf11_s3']].to_numpy()
    
    assigned_rent_for_train = []
    for i in range(len(df_train)):
        options = rf_options_train[i] # ['everyday', 'party', 'work']
        scores = rf_scores_train[i]  # [0.5, 0.2, 0.0]
        
        # [S1] 점수가 0보다 큰(>) 유효한 옵션만 필터링
        valid_options = [opt for opt, score in zip(options, scores) if score > 0]
        
        # (All-Zero는 DB에서 미리 걸렀으므로 valid_options는 최소 1개 보장됨)
        assigned_rent_for_train.append(np.random.choice(valid_options))

    df_train['rented for'] = assigned_rent_for_train
    print("'rented for' 컬럼 값 교체 완료 (유사도 0 제외).")
    
    # 3j. 컬럼 순서 재배치
    all_cols_train = df_train.columns.tolist()
    other_cols_train = [col for col in all_cols_train if col not in ['product_code', 'prod_name']]
    final_cols_train = ['product_code', 'prod_name'] + other_cols_train
    df_train = df_train[final_cols_train]
    print("훈련 데이터 컬럼 순서 재배치 완료.")

    # 3k. 파일 저장
    df_train.to_csv(out_train_file, index=False, encoding='utf-8-sig')
    print(f"✅ 훈련 데이터 처리 완료 -> {out_train_file}")


    # --- 4. 테스트 데이터 (Test Set) 처리 (ModCloth) ---
    print(f"\n--- 3. 테스트 데이터 ({file_modcloth}) 처리 ---")
    
    # 4a. 로드 및 'dresses' 카테고리 필터링
    df_test_raw = pd.read_csv(file_modcloth)
    df_test = df_test_raw[df_test_raw['category'] == 'dresses'].copy() # 'dresses' (복수형)
    print(f"원본 {len(df_test_raw):,}건 -> 'dresses' {len(df_test):,}건 필터링.")
    
    # 4b. 테스트 데이터 단위 통일
    print("Test: 단위 통일 시작...")
    
    # [S2 기능] 4b-1. Rating (ModCloth 'quality' 0-5 -> 'rating' 0-10으로 보정)
    if 'quality' in df_test.columns:
        df_test['quality'] = pd.to_numeric(df_test['quality'], errors='coerce') * 2.0
        df_test['quality'] = df_test['quality'].clip(0, 10)
        df_test = df_test.rename(columns={'quality': 'rating'})
        print("  Test: 'quality' (0-5) -> 'rating' (0-10) 스케일 보정 완료.")
    elif 'rating' in df_test.columns:
         df_test['rating'] = pd.to_numeric(df_test['rating'], errors='coerce') * 2.0
         df_test['rating'] = df_test['rating'].clip(0, 10)
         print("  Test: 'rating' (0-5) -> 'rating' (0-10) 스케일 보정 완료.")
    else: 
        print("  Test: 'quality' 또는 'rating' 컬럼을 찾을 수 없어 표준화를 건너뜁니다.")
    
    # 4b-2. Height (cm로 변환, 결측치 제거)
    df_test['height'] = df_test['height'].apply(convert_height_to_cm)
    df_test = df_test.dropna(subset=['height']) 
    df_test = df_test.rename(columns={'height': 'height_cm'}) 
    print(f"  Test: Height -> cm 변환 및 결측치 제거. (남은 {len(df_test):,}건)")
    
    # [S2 + 오류 방지] 4b-3. Weight (lbs -> kg) (Test는 이 컬럼이 없을 수 있음)
    if 'weight' in df_test.columns:
        df_test['weight'] = df_test['weight'].apply(convert_weight_to_kg)
        df_test = df_test.rename(columns={'weight': 'weight_kg'})
        print("  Test: Weight -> kg 변환 완료.")
    else: 
        print("  Test: 'weight' 컬럼을 찾을 수 없어 표준화를 건너뜁니다.")

    # 4b-4. Bust Size (결측치 'unknown' 처리)
    df_test['bra size'] = df_test['bra size'].astype(float).astype('Int64').astype(str).replace('<NA>', 'unknown')
    df_test['cup size'] = df_test['cup size'].fillna('unknown').apply(lambda x: str(x).lower().replace('+', '').strip())
    print("  Test: 'bra size', 'cup size' 정제/결측치 처리 완료.")
    
    # [S2 기능] 4b-5. Clothing Size (0-10 정규화 - *Train 기준* 적용)
    if 'size' in df_test.columns:
        df_test['size_numeric'] = df_test['size'].apply(convert_size_to_numeric)
        df_test = df_test.dropna(subset=['size_numeric'])

        if (max_size - min_size) == 0:
            df_test['size_scaled_0_to_10'] = 5.0 
        else:
            df_test['size_scaled_0_to_10'] = 10 * (df_test['size_numeric'] - min_size) / (max_size - min_size)
        
        df_test['size_scaled_0_to_10'] = df_test['size_scaled_0_to_10'].clip(0, 10)
            
        df_test = df_test.drop(columns=['size', 'size_numeric'])
        print(f"  Test: 'size' -> 'size_scaled_0_to_10' (Train 기준 0-10점) 정규화 완료. (남은 {len(df_test):,}건)")
    else:
        print("  Test: 'size' 컬럼을 찾을 수 없어 표준화를 건너뜁니다.")
    
    # 4c. item_id -> product_code 컬럼명 변경
    df_test = df_test.rename(columns={'item_id': 'product_code'})
    
    # 4d. 60% H&M DB (샘플) 생성 (유사도 0점 제외된 DB 기준)
    hm_dresses_db_60pct = hm_dresses_db_100pct.sample(frac=0.6, random_state=42)
    hm_codes_60pct = hm_dresses_db_60pct['product_code'].values
    print(f"H&M 드레스 60% 샘플 목록 생성: {len(hm_codes_60pct):,}개 (원본의 60%)")
    
    # 4e. product_code 값을 60% H&M DB에서 랜덤 할당 (교체)
    assigned_codes_test = np.random.choice(hm_codes_60pct, size=len(df_test))
    df_test['product_code'] = assigned_codes_test
    print(f"{len(df_test):,}개 레코드에 60% H&M product_code 랜덤 할당 완료.")

    # 4f. 할당된 product_code에 매핑되는 *모든 정보* 조회
    assigned_data_test = hm_dresses_db_indexed.loc[assigned_codes_test]

    # 4g. 'prod_name' 컬럼 신규 생성
    df_test['prod_name'] = assigned_data_test['prod_name'].values
    print("'prod_name' 컬럼 신규 생성 완료.")

    # [S1 기능] 4h. 'rented for' 컬럼을 (유사도 > 0 인) Top-3 라벨 중 하나로 '신규 생성'
    rf_options_test = assigned_data_test[['rf11_top1', 'rf11_top2', 'rf11_top3']].to_numpy()
    rf_scores_test = assigned_data_test[['rf11_s1', 'rf11_s2', 'rf11_s3']].to_numpy()
    
    assigned_rent_for_test = []
    for i in range(len(df_test)):
        options = rf_options_test[i]
        scores = rf_scores_test[i]
        valid_options = [opt for opt, score in zip(options, scores) if score > 0]
        assigned_rent_for_test.append(np.random.choice(valid_options))

    df_test['rented for'] = assigned_rent_for_test
    print("'rented for' 컬럼 신규 생성 및 값 할당 완료 (유사도 0 제외).")
    
    # 4i. 컬럼 순서 재배치
    all_cols_test = df_test.columns.tolist()
    other_cols_test = [col for col in all_cols_test if col not in ['product_code', 'prod_name']]
    final_cols_test = ['product_code', 'prod_name'] + other_cols_test
    df_test = df_test[final_cols_test]
    print("테스트 데이터 컬럼 순서 재배치 완료.")

    # 4j. 파일 저장
    df_test.to_csv(out_test_file, index=False, encoding='utf-8-sig')
    print(f"✅ 테스트 데이터 처리 완료 -> {out_test_file}")

    print("\n--- 모든 작업 완료 ---")

except FileNotFoundError as e:
    print(f"[파일 로드 오류] {e}.")
    print("필요한 파일 5개(renttherunway, modcloth, articles, hm_articles..., renttherunway_filled)가 스크립트와 같은 폴더에 있는지 확인하세요.")
except KeyError as e:
    print(f"[KeyError] {e}.")
    print("원본 CSV 파일에 'item_id', 'category', 'bust size', 'height', 'weight', 'rating'/'quality', 'size' 등의 필수 컬럼명이 정확히 존재하는지 확인하세요.")
except Exception as e:
    print(f"처리 중 예기치 않은 오류 발생: {e}")
    traceback.print_exc()