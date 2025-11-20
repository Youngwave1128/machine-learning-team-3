# -*- coding: utf-8 -*-
"""
articles_top3_11cats.py

기능:
- 같은 폴더의 articles.csv만 사용
- product_code 기준 중복 제거
- description(detail_desc) 텍스트(비면 보조 필드로 보강)만으로
  rent_for 11개 카테고리와 TF-IDF 코사인 유사도로 Top-3 라벨링
- 결과 CSV(개별 Top-3) + 집계 CSV(Top-3 '등장 여부' 기준) 저장

사용 예:
    python articles_top3_11cats.py \
      --src ./articles.csv \
      --out_items ./hm_articles_rentfor_top3_11cats.csv \
      --out_counts ./hm_counts_top3_presence_11cats.csv
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1) 카테고리 11개 정의
# =========================
def get_category_defs() -> List[Tuple[str, str]]:
    """
    반환: [(라벨, 설명문)]
    설명문은 TF-IDF 코사인에서 '시드 키워드'로 작동.
    """
    return [
        # [1] Party
        ("party_specific",
         "holiday party, birthday party, christmas party, new years party, "
         "bachelorette party; office party, company party; date-specific or seasonal party"),
        ("party_general",
         "general party without specific holiday terms; friends gathering, club, night out"),

        # [2] Wedding
        ("wedding_black_tie",
         "black tie wedding; very formal wedding; tuxedo; floor-length gown; elegant evening"),
        ("wedding_general",
         "wedding guest; seasonal or venue terms like summer wedding, outdoor wedding, fall wedding"),

        # [3] Date
        ("date_specific",
         "special occasion date such as birthday dinner, anniversary dinner, valentines day"),
        ("date_general",
         "general date night or dinner date; casual romantic night out"),

        # [4] Other (신규 세분)
        ("photoshoot",
         "photoshoot or photography session such as engagement photos, portrait session"),

        # 단일 라벨
        ("formal_affair", "formal affair or formal event; gala; ceremony; reception"),
        ("work",          "work or office wear; business casual; professional outfit"),
        ("everyday",      "everyday casual; daily wear; simple comfortable basics"),
        ("vacation",      "vacation; resort; holiday travel outfit; beach trip"),
    ]


# =========================
# 2) 유틸 함수
# =========================
def normalize_col(name: str) -> str:
    """컬럼명을 소문자+언더스코어로 통일."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col(c) for c in df.columns]
    return out

def build_text_from_row(row: pd.Series, text_cols: List[str]) -> str:
    """
    detail_desc를 우선으로 하고, 비거나 없으면 보조 필드들을 공백으로 이어 붙임.
    """
    parts = []
    # detail_desc 우선
    if "detail_desc" in row.index and isinstance(row["detail_desc"], str) and row["detail_desc"].strip():
        parts.append(row["detail_desc"].strip())
    # 보조 필드
    for c in text_cols:
        if c == "detail_desc":
            continue
        v = row.get(c, "")
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip().lower()


# =========================
# 3) 파이프라인
# =========================
def run(src: Path, out_items: Path, out_counts: Path) -> None:
    # ── 데이터 로드
    if not src.exists():
        raise FileNotFoundError(f"articles.csv를 찾을 수 없습니다: {src}")
    df = pd.read_csv(src, on_bad_lines="skip", low_memory=False)
    df = normalize_columns(df)

    # ── product_code 중복 제거
    if "product_code" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["product_code"], keep="first").copy()
        after = len(df)
        print(f"[INFO] product_code dedup: {before:,} -> {after:,}")
    else:
        print("[WARN] 'product_code' 컬럼 없음 → 중복 제거 생략")

    # ── 설명 텍스트 생성: detail_desc 우선, 비면 보조 필드 보강
    text_cols = [c for c in [
        "detail_desc", "prod_name", "product_type_name", "product_group_name",
        "graphical_appearance_name", "colour_group_name",
        "perceived_colour_value_name", "perceived_colour_master_name",
        "department_name", "index_name", "index_group_name",
        "section_name", "garment_group_name"
    ] if c in df.columns]

    if "detail_desc" not in text_cols:
        # detail_desc가 없으면 보조 필드만으로 구성
        print("[WARN] 'detail_desc' 없음 → 보조 필드만으로 description 구성")

    df["_desc_all"] = df.apply(lambda r: build_text_from_row(r, text_cols), axis=1)

    # 식별자 컬럼
    article_ids = df["article_id"].values if "article_id" in df.columns else np.arange(len(df))
    prod_codes = df["product_code"].values if "product_code" in df.columns else [""] * len(df)

    docs = pd.DataFrame({"article_id": article_ids, "product_code": prod_codes, "_desc_all": df["_desc_all"]})

    # ── 카테고리 11개 + TF-IDF 코사인 Top-3
    cat_defs = get_category_defs()
    labels = [c[0] for c in cat_defs]
    descs  = [c[1] for c in cat_defs]

    # 코퍼스: [카테고리 설명 11개] + [상품 설명 N개]
    corpus = descs + docs["_desc_all"].tolist()

    # TF-IDF: 희귀 키워드도 반영되도록 min_df=1, 너무 흔한 건 억제
    vec = TfidfVectorizer(
        ngram_range=(1, 2),   # 1~2-gram
        min_df=1,
        max_df=0.98,
        stop_words="english"
    )
    X = vec.fit_transform(corpus)

    n_cat = len(descs)    # = 11
    X_cat = X[:n_cat]
    X_doc = X[n_cat:]

    # 코사인 유사도 → Top-3
    S = cosine_similarity(X_doc, X_cat)          # (N, 11)
    top_idx = np.argsort(-S, axis=1)[:, :3]
    top_scores = np.take_along_axis(S, top_idx, axis=1)
    top_labels = np.array(labels)[top_idx]

    labeled = docs.copy()
    labeled["rf11_top1"] = top_labels[:, 0]
    labeled["rf11_s1"]   = top_scores[:, 0]
    labeled["rf11_top2"] = top_labels[:, 1]
    labeled["rf11_s2"]   = top_scores[:, 1]
    labeled["rf11_top3"] = top_labels[:, 2]
    labeled["rf11_s3"]   = top_scores[:, 2]

    # ── 결과 저장(개별 Top-3)
    out_items.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_items, index=False, encoding="utf-8-sig")

    # ── 집계: “Top-3에 등장 여부” 기준(Top-1만 아님)
    present = {lab: 0 for lab in labels}
    for _, row in labeled.iterrows():
        labs = {row["rf11_top1"], row["rf11_top2"], row["rf11_top3"]}
        for lab in labs:
            present[lab] += 1
    counts = pd.DataFrame([{"rent_for": k, "item_count_top3_presence": v} for k, v in present.items()])
    counts = counts.sort_values("rent_for").reset_index(drop=True)

    counts.to_csv(out_counts, index=False, encoding="utf-8-sig")

    # ── 로그
    print(f"[INFO] 총 아이템 수(중복 제거 후): {len(labeled):,}")
    print(f"[INFO] 라벨(11): {labels}")
    print(f"[INFO] Top-3 라벨링 저장 → {out_items}")
    print(f"[INFO] Top-3 존재 집계 저장 → {out_counts}")
    print(counts.head(11))


# =========================
# 4) CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="articles.csv → rent_for 11종 Top-3 코사인 매칭 & 집계")
    p.add_argument("--src", type=str, default="articles.csv", help="입력 CSV 경로")
    p.add_argument("--out_items", type=str, default="hm_articles_rentfor_top3_11cats.csv",
                   help="개별 Top-3 결과 CSV 경로")
    p.add_argument("--out_counts", type=str, default="hm_counts_top3_presence_11cats.csv",
                   help="Top-3 존재 집계 CSV 경로")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.src), Path(args.out_items), Path(args.out_counts))
