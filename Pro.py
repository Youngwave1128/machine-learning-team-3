import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# NLTK 불용어 다운로드 (네트워크 오류 시 Fallback)
try:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print("NLTK stopwords loaded successfully.")
    # N-gram 분석에 방해가 될 수 있는 일반적인 단어 추가
    # (이 단어들은 제외하고 구체적인 '이벤트' 키워드를 찾기 위함)
    stop_words.update(['dress', 'fit', 'size', 'wear', 'wore', 'true', 'great', 
                       'perfect', 'comfortable', 'loved', 'feel', 'felt', 'little',
                       'really', 'got', 'like', 'made', 'small', 'large', 'would', 
                       'lbs', 'also', 'bit', 'even', 'run', 'runs', 'top', 
                       'get', 'one', 'much', 'ordered', 'rented'])
except Exception as e:
    print(f"Warning: Failed to download/load NLTK stopwords ({e}). Using a minimal fallback list.")
    # 수동으로 최소한의 불용어 리스트 정의
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
                       'dress', 'fit', 'size', 'wear', 'wore', 'true', 'great', 'perfect', 'comfortable', 'loved', 'feel', 'felt', 'little', 'really', 'got', 'like', 'made', 'small', 'large', 'would', 'lbs'])


# 1. 데이터 로드 및 기본 전처리
try:
    # ParserError 수정을 위해 on_bad_lines='skip' 추가
    df = pd.read_csv('renttherunway_final_data.csv', on_bad_lines='skip')
    print(f"데이터 로드 완료. 총 {len(df)}개 행.")
except FileNotFoundError:
    print("오류: 'renttherunway_final_data.csv' 파일을 찾을 수 없습니다.")
    exit()

# 'rented for'와 'review_text'에 결측치가 있는 행 제거
df_cleaned = df.dropna(subset=['rented for', 'review_text']).copy()
print(f"결측치 제거 후 {len(df_cleaned)}개 행.")

# 2. 텍스트 정제 함수 정의
def preprocess_text(text):
    """
    텍스트를 소문자로 변환하고, 구두점을 제거하며, 숫자도 제거합니다.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # 소문자 변환
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)      # 숫자 제거
    return text

# 3. N-gram 빈도 분석 함수
def get_top_ngrams(texts, ngram_range=(2, 3), top_n=30):
    """
    주어진 텍스트 리스트에서 상위 N-gram을 추출합니다.
    """
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, 
                                     stop_words=list(stop_words), 
                                     max_features=5000)
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    sum_words = X.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_n]

# 4. 분석 실행
# 'rented for'의 모든 고유 카테고리 추출
categories_to_analyze = df_cleaned['rented for'].unique()

print(f"\n--- 총 {len(categories_to_analyze)}개 카테고리에 대한 N-gram 분석 시작 ---")
print(f"분석 대상: {list(categories_to_analyze)}")

analysis_results = {}

for category in categories_to_analyze:
    print(f"\nProcessing: '{category}'")
    
    # 1. 해당 카테고리 데이터 필터링 및 정제
    cat_texts = df_cleaned[df_cleaned['rented for'] == category]['review_text'].apply(preprocess_text)
    
    if cat_texts.empty:
        print(f"'{category}'에 대한 텍스트가 없습니다.")
        continue

    # 2. N-gram 추출 (2~3단어 조합)
    top_ngrams = get_top_ngrams(cat_texts, ngram_range=(2, 3), top_n=30)
    
    analysis_results[category] = top_ngrams

print("\n--- 분석 완료 ---")

# 5. 결과 출력
print("\n### 전체 카테고리 N-gram 빈도 분석 결과 ###")
print("=" * 40)
print("이 키워드들을 바탕으로 2-way 분류가 유의미한 카테고리를 선별할 수 있습니다.")

for category, ngrams in analysis_results.items():
    print(f"\n[ {category.upper()} ] Top 30 N-grams:")
    print("-" * (len(category) + 20))
    if ngrams:
        # 결과를 2열로 보기 좋게 출력
        output_lines = [f"  {freq:4d} : {ngram}" for ngram, freq in ngrams]
        mid_point = (len(output_lines) + 1) // 2
        col1 = output_lines[:mid_point]
        col2 = output_lines[mid_point:]
        
        for i in range(mid_point):
            col1_item = col1[i]
            col2_item = col2[i] if i < len(col2) else ""
            print(f"{col1_item:<35} | {col2_item}")
    else:
        print("  추출된 N-gram이 없습니다.")
        
print("\n" + "=" * 40)
