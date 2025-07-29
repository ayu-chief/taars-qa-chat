import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import html
import re

# ---------------------
# データ読み込み（日本語列名に対応）
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "お問い合わせ内容": "質問",
        "返信内容": "回答"
    })
    df = df[["質問", "回答", "ジャンル"]].dropna(subset=["質問", "回答"])
    return df

# ---------------------
# マスキングリスト読み込み
# ---------------------
@st.cache_data
def load_masking_lists():
    df_pm = pd.read_excel("PM担当者一覧.xlsx")
    df_pm.columns = df_pm.columns.str.strip()
    pm_names = set(df_pm.iloc[:, 0].astype(str) + df_pm.iloc[:, 1].astype(str))

    df_buildings = pd.read_excel("物件一覧.xlsx")
    df_buildings.columns = df_buildings.columns.str.strip()
    building_names = set(df_buildings.iloc[:, 0].astype(str))
    
    return pm_names, building_names

# ---------------------
# マスキング処理
# ---------------------
def mask_text(text, pm_names, building_names):
    for name in pm_names:
        text = text.replace(name, "〇〇さん")
    for bname in building_names:
        text = text.replace(bname, "〇〇物件")
    return text

# ---------------------
# モデルと埋め込み準備
# ---------------------
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["質問"].tolist(), convert_to_tensor=True)
    return model, embeddings

# ---------------------
# 回答検索
# ---------------------
def search_answer(query, model, corpus_embeddings, df, pm_names, building_names, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_indices = cos_scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        q = html.escape(df.iloc[idx]["質問"])
        a = html.escape(df.iloc[idx]["回答"])
        q = mask_text(q, pm_names, building_names)
        a = mask_text(a, pm_names, building_names)
        results.append({"質問": q, "回答": a})
    return results

# ---------------------
# メインページ：チャット形式検索
# ---------------------
def main_page(df, model, corpus_embeddings, pm_names, building_names):
    st.title("【TAARS】FAQ検索チャット")
    st.markdown("過去のFAQから似た質問と回答を検索できます")

    st.markdown("#### 入力例：")
    st.markdown("- 契約書を再発行したい\n- 物件の確認方法\n- 担当者に連絡したい")

    user_input = st.text_input("質問を入力してください", placeholder="例：契約書を再発行したい")
    st.markdown('<span style="font-size: 0.9em; color: gray;">結果が多いため、<strong>質問を簡潔に</strong>すると絞り込みやすくなります。</span>', unsafe_allow_html=True)

    if st.button("検索") or user_input:
        with st.spinner("検索中..."):
            results = search_answer(user_input, model, corpus_embeddings, df, pm_names, building_names)
            for r in results:
                st.markdown("#### Q: " + r["質問"])
                st.markdown("**A:** " + r["回答"])
                st.markdown("---")

# ---------------------
# サイドバー・ジャンル別ページ
# ---------------------
def genre_page(df, pm_names, building_names):
    st.title("ジャンル別FAQ一覧")

    genres = sorted(df["ジャンル"].dropna().unique())
    selected_genre = st.selectbox("ジャンルを選択してください", genres)

    filtered_df = df[df["ジャンル"] == selected_genre]

    for _, row in filtered_df.iterrows():
        question = mask_text(html.escape(row["質問"]), pm_names, building_names)
        answer = mask_text(html.escape(row["回答"]), pm_names, building_names)
        st.markdown("#### Q: " + question)
        st.markdown("**A:** " + answer)
        st.markdown("---")

# ---------------------
# アプリ実行
# ---------------------
def main():
    df = load_data()
    pm_names, building_names = load_masking_lists()
    model, corpus_embeddings = load_model_and_embeddings(df)

    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio("ページ選択", ["FAQ検索チャット", "ジャンル別FAQ一覧"])

    if page == "FAQ検索チャット":
        main_page(df, model, corpus_embeddings, pm_names, building_names)
    else:
        genre_page(df, pm_names, building_names)

if __name__ == "__main__":
    main()
