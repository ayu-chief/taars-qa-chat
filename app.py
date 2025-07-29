import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# データ読み込みとマスキング
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")

    # 列名の空白や改行を除去（日本語列名対応）
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)

    # 日本語→内部用にリネーム
    df = df.rename(columns={
        "お問い合わせ内容": "質問",
        "返信内容": "回答",
        "ジャンル": "ジャンル"
    })

    # 必要列のみ抽出
    df = df[["質問", "回答", "ジャンル"]].dropna(subset=["質問", "回答"])
    return df

@st.cache_data
def load_masking_lists():
    df_pm = pd.read_excel("PM担当者一覧.xlsx")
    df_pm.columns = df_pm.columns.str.replace(r"\s+", "", regex=True)
    pm_names = set(df_pm["姓"].astype(str)) | set(df_pm["名"].astype(str))

    df_bld = pd.read_excel("物件一覧.xlsx")
    df_bld.columns = df_bld.columns.str.replace(r"\s+", "", regex=True)
    building_names = set(df_bld["物件名"].astype(str))

    return pm_names, building_names

def mask_text(text, pm_names, building_names):
    for name in sorted(pm_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "〇〇さん", text)
    for name in sorted(building_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "〇〇物件", text)
    return text

# ---------------------------
# 類似質問検索
# ---------------------------
@st.cache_resource
def load_model_and_embeddings(df):
    model = TfidfVectorizer()
    embeddings = model.fit_transform(df["質問"].tolist())
    return model, embeddings

def search_similar_questions(query, model, embeddings, df, top_k=3):
    query_vec = model.transform([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0:
            results.append({
                "score": round(float(score), 2),
                "質問": df.iloc[idx]["質問"],
                "回答": df.iloc[idx]["回答"]
            })
    return results

# ---------------------------
# メインページ：類似QA検索
# ---------------------------
def show_chat_page(df, model, embeddings, pm_names, building_names):
    st.subheader("質問を入力してください")
    example_expander = st.expander("入力例を見る")
    with example_expander:
        st.markdown("""
        - 契約書を再発行したい  
        - 物件の確認方法  
        - 担当者に連絡したい  
        """)

    query = st.text_input("")

    if query:
        with st.spinner("検索中..."):
            results = search_similar_questions(query, model, embeddings, df)

        if results:
            st.success("こちらの内容が近いかもしれません：")
            for r in results:
                masked_q = mask_text(r["質問"], pm_names, building_names)
                masked_a = mask_text(r["回答"], pm_names, building_names)
                with st.container():
                    st.markdown(f"**Q: {masked_q}**")
                    st.markdown(f"A: {masked_a}")
                    st.markdown("---")
        else:
            st.warning("類似する質問が見つかりませんでした。")

# ---------------------------
# サイドページ：ジャンル別一覧
# ---------------------------
def show_genre_page(df, pm_names, building_names):
    st.subheader("ジャンル別 FAQ")

    genre_list = sorted(df["ジャンル"].dropna().unique())
    selected_genre = st.selectbox("ジャンルを選択", ["すべて"] + genre_list)

    if selected_genre == "すべて":
        filtered = df
    else:
        filtered = df[df["ジャンル"] == selected_genre]

    if filtered.empty:
        st.info("このジャンルには質問が登録されていません。")
        return

    for i, row in filtered.iterrows():
        masked_q = mask_text(row["質問"], pm_names, building_names)
        masked_a = mask_text(row["回答"], pm_names, building_names)
        with st.container():
            st.markdown(f"**Q: {masked_q}**")
            st.markdown(f"A: {masked_a}")
            st.markdown("---")

# ---------------------------
# サイドバーとルーティング
# ---------------------------
def main():
    st.title("【TAARS】FAQ検索チャット")
    st.caption("過去のFAQから似た質問と回答を検索できます")

    page = st.sidebar.radio("表示ページを選んでください", ["類似QA検索チャット", "ジャンル別FAQ一覧"])

    df = load_data()
    pm_names, building_names = load_masking_lists()
    model, embeddings = load_model_and_embeddings(df)

    if page == "類似QA検索チャット":
        show_chat_page(df, model, embeddings, pm_names, building_names)
    else:
        show_genre_page(df, pm_names, building_names)

if __name__ == "__main__":
    main()
