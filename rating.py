import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 1️⃣ CSV 불러오기 (전처리된 데이터)
df = pd.read_csv("preprocessed_renttherunway.csv")

# 2️⃣ HuggingFace의 감정 분석 모델 로드
#  - RoBERTa 기반, 긍정/중립/부정 확률 출력
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 3️⃣ 리뷰 텍스트 → 감정 점수 → 0~10 평점으로 변환하는 함수
def predict_sentiment_rating(text):
    # 리뷰가 없거나 NaN이면 None 반환
    if pd.isna(text) or str(text).strip() == "":
        return None

    # 텍스트 토큰화 후 모델 입력
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 추론 (forward pass)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()

    # 확률 분포: [negative, neutral, positive]
    labels = ['negative', 'neutral', 'positive']
    sentiment = dict(zip(labels, probs))

    # (positive - negative) = 감정 극성 (-1~+1)
    sentiment_score = sentiment['positive'] - sentiment['negative']

    # 0~10 스케일로 변환
    rating_0_to_10 = round((sentiment_score + 1) * 5, 2)

    return rating_0_to_10

# 4️⃣ rating이 비어 있고 review_text가 존재하는 행만 예측
missing_mask = df["rating"].isna() & df["review_text"].notna()
df["predicted_rating"] = None

print(f"🧩 예측할 리뷰 수: {missing_mask.sum()}개")
for i in tqdm(df[missing_mask].index, desc="Predicting missing ratings (0~10 scale)"):
    df.loc[i, "predicted_rating"] = predict_sentiment_rating(df.loc[i, "review_text"])

# 5️⃣ 기존 rating은 유지하고, 빈 칸만 predicted로 채움
df["rating_filled"] = df["rating"].combine_first(df["predicted_rating"])

# 6️⃣ 결과 저장
df.to_csv("renttherunway_filled_sentiment.csv", index=False, encoding="utf-8-sig")
print("✅ 감정 분석으로 rating의 빈값 자동 채우기 완료!")
print("📁 결과 파일: renttherunway_filled_sentiment.csv")
