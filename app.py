import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")
st.title("【TAARS】FAQ検索チャット")
st.markdown("質問を入力すると、過去のFAQから近いものを提案します。")

# 入力例を常に表示
st.markdown("💡 **入力例**：")
st.markdown("- ログインできない")
st.markdown("- 支払い方法を教えてください")
st.markdown("- 契約申請について")

@st.cache_data
def load_data():
    return pd.read_csv("qa_data.csv", encoding="utf-8")

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

def format_conversation(text):
    # サポート対応や氏名ごとのやり取りを見やすく整形（例：「〇〇様」「[サポート]」など）
    text = re.sub(r"(\n)?(\[サポート\]|[\w\s]+様)", r"\n---\n\2", text)
    return text.strip()

df = load_data()
model, corpus_embeddings = load_model_and_embeddings(df)

user_input = st.text_input("❓ 質問を入力してください", "")

if user_input:
    with st.spinner("検索中..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]

        # 類似度0.5以上のみ表示
        filtered_hits = [hit for hit in results if hit["score"] >= 0.5]
        num_hits = len(filtered_hits)

        if num_hits == 0:
            st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
        else:
            if num_hits > 10:
                st.info(f"{num_hits} 件見つかりました。質問をより具体的に入力すると、結果が絞り込まれます。")

            st.markdown("### 🔍 類似するQA：")
            for hit in filtered_hits:
                row = df.iloc[hit["corpus_id"]]
                st.markdown(f"**{row['question']}**")
                with st.expander("▶ 回答を見る"):
                    formatted = format_conversation(str(row['answer']))
                    st.markdown(formatted.replace("\n", "  \n"))  # Streamlitでの改行表示
                st.markdown("---")
