import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")
st.title("【TAARS】FAQ検索チャット")
st.markdown("質問を入力すると、過去のFAQから近いものを提案します。")

# 入力例（行間を詰めて表示）
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

# 入力欄の見出しを強調
st.markdown("### ❓ **質問を入力してください**")

# 初期表示件数
if "visible_count" not in st.session_state:
    st.session_state.visible_count = 10

@st.cache_data
def load_data():
    return pd.read_csv("qa_data.csv", encoding="utf-8")

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

def format_conversation(text):
    # 会話整形（サポート／ユーザーごとに絵文字とラベル付け）
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        if "[サポート]" in line:
            line = line.replace("[サポート]", "💬 **サポート：**")
        elif "[ユーザー]" in line:
            line = line.replace("[ユーザー]", "👤 **ユーザー：**")
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

# データとモデルの読み込み
df = load_data()
model, corpus_embeddings = load_model_and_embeddings(df)

# ユーザーの質問入力
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
            st.markdown("### 🔍 類似するQA：")
            st.info("💬 はサポート、👤 はユーザーの発言を表しています。")

            for hit in filtered_hits[:st.session_state.visible_count]:
                row = df.iloc[hit["corpus_id"]]
                st.markdown(f"**{row['question']}**")
                with st.expander("▶ 回答を見る"):
                    formatted = format_conversation(str(row['answer']))
                    st.markdown(formatted.replace("\n", "  \n"))
                st.markdown("---")

            if st.session_state.visible_count < num_hits:
                if st.button("🔽 もっと表示する"):
                    st.session_state.visible_count += 10
                    st.rerun()
else:
    # 新規入力時はカウンターをリセット
    st.session_state.visible_count = 10
