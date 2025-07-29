import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")
st.title("【TAARS】FAQ検索チャット")
st.markdown("質問を入力すると、過去のFAQから近いものを提案します。")

# 入力例：CSSで行間を詰めた表示
st.markdown("""
<style>
ul.input-examples { margin-top: 0.2rem; margin-bottom: 1rem; line-height: 1.2; padding-left: 1.2rem; }
</style>
💡 **入力例**：
<ul class="input-examples">
<li>ログインできない</li>
<li>支払い方法を教えてください</li>
<li>契約申請について</li>
</ul>
""", unsafe_allow_html=True)

# 強調された入力フィールド見出し
st.markdown("### ❓ **質問を入力してください**")

@st.cache_data
def load_data():
    return pd.read_csv("qa_data.csv", encoding="utf-8")

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

def format_conversation(text):
    # 会話風の整形：サポートとユーザーの発言に絵文字と改行を入れる
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        if "[サポート]" in line:
            line = line.replace("[サポート]", "💬 **サポート：**")
        elif "[ユーザー]" in line:
            line = line.replace("[ユーザー]", "👤 **ユーザー：**")
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

df = load_data()
model, corpus_embeddings = load_model_and_embeddings(df)

user_input = st.text_input("", "")

if user_input:
    with st.spinner("検索中..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]

        filtered_hits = [hit for hit in results if hit["score"] >= 0.5]
        num_hits = len(filtered_hits)

        if num_hits == 0:
            st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
        else:
            if num_hits > 10:
                st.info(f"{num_hits} 件見つかりました。質問をより具体的に入力すると、結果が絞り込まれます。")

            st.markdown("### 🔍 類似するQA：")
            st.info("💬 はサポート、👤 はユーザーの発言を表しています。")  # ← ここが凡例の追加

            for hit in filtered_hits:
                row = df.iloc[hit["corpus_id"]]
                st.markdown(f"**{row['question']}**")
                with st.expander("▶ 回答を見る"):
                    formatted = format_conversation(str(row['answer']))
                    st.markdown(formatted.replace("\n", "  \n"))
                st.markdown("---")
