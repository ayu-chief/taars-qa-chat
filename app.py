
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")
st.title("【TAARS】FAQ検索チャット")
st.markdown("質問を入力すると、過去のFAQから近いものを提案します。")

with st.expander("💡 入力例（クリックして表示）"):
    st.markdown("- 例1：ログインできない\n- 例2：支払い方法を教えてください\n- 例3：契約申請について")

@st.cache_data
def load_data():
    return pd.read_csv("qa_data.csv", encoding="utf-8")

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

df = load_data()
model, corpus_embeddings = load_model_and_embeddings(df)

user_input = st.text_input("❓ 質問を入力してください", "")

if user_input:
    with st.spinner("検索中..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]

        # スコアが一定以上（0.5）だけを抽出
        filtered_hits = [hit for hit in results if hit["score"] >= 0.5]
        num_hits = len(filtered_hits)

        if num_hits == 0:
            st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
        else:
            if num_hits > 10:
                st.info(f"{num_hits} 件見つかりました。検索結果が多いため、質問内容をさらに詳細に入力することで絞り込めます。")

            st.markdown("### 🔍 類似するQA：")
            for hit in filtered_hits:
                row = df.iloc[hit["corpus_id"]]
                st.markdown(f"**{row['question']}**")
                with st.expander("▶ 回答を見る"):
                    st.markdown(f"{row['answer']}")
                st.markdown("---")
