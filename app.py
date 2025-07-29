
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")
st.title("【TAARS】FAQ検索チャット")
st.markdown("質問を入力すると、過去のFAQから近いものを提案します。")

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
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
        st.markdown("### 🔍 類似するQA：")
        for i, hit in enumerate(hits):
            row = df.iloc[hit["corpus_id"]]
            st.markdown(f"**Q{i+1}: {row['question']}**")
            st.markdown(f"**A{i+1}: {row['answer']}**")
            st.markdown("---")
