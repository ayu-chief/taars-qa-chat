import streamlit as st
import pandas as pd
import hashlib
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")

# カスタムCSS
st.markdown("""
<style>
body {
    background-color: #f4f8f9;
}
h1, h2, h3 {
    color: #004d66;
}
div.stButton > button {
    background-color: #00838f;
    color: white;
}
.qa-box {
    background-color: white;
    border-left: 5px solid #a2d7c7;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 4px rgba(0,0,0,0.05);
}
.support-bubble {
    background-color: #e1f0f9;
    padding: 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.25rem;
}
.user-bubble {
    background-color: #f1e8f9;
    padding: 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.25rem;
}
.info-banner {
    background-color: #d8ecf0;
    padding: 1rem;
    border-radius: 6px;
    margin-bottom: 2rem;
    color: #003d33;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.markdown("""
<div style='background-color: #e3f3ec; padding: 2rem 1rem; border-radius: 6px; text-align: center;'>
    <h1 style='color: #004d66;'>【TAARS】FAQ検索チャット</h1>
    <p style='font-size: 1.1rem;'>過去のFAQから似た質問と回答を検索できます</p>
</div>
""", unsafe_allow_html=True)

# 入力欄
st.markdown("### 質問を入力してください")
user_input = st.text_input("", "")

# 初期状態管理
if "visible_count" not in st.session_state:
    st.session_state.visible_count = 10

# CSVとモデル読み込み
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

# 安全な key 作成
def make_safe_key(text, i):
    return f"k_{hashlib.md5((text + str(i)).encode()).hexdigest()}"

# 会話整形
def format_conversation(text):
    lines = text.splitlines()
    formatted = ""
    for line in lines:
        if "[サポート]" in line:
            line = line.replace("[サポート]", '<div class="support-bubble">💬 サポート')
            formatted += line + "</div>\n"
        elif "[ユーザー]" in line:
            line = line.replace("[ユーザー]", '<div class="user-bubble">👤 ユーザー')
            formatted += line + "</div>\n"
        else:
            formatted += f"<div>{line}</div>\n"
    return formatted

# 検索処理
if user_input:
    with st.spinner("検索中..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]
        filtered_hits = [hit for hit in results if hit["score"] >= 0.5]
        num_hits = len(filtered_hits)
        st.session_state.visible_count = 10  # reset on each search

        if num_hits == 0:
            st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
        else:
            st.success(f"{num_hits} 件の結果が見つかりました。")
            if num_hits > 10:
                st.info("結果が多いため、質問をさらに具体的にすると絞り込みやすくなります。")

            st.markdown("<div class='info-banner'>💬 はサポート、👤 はユーザーの発言を表しています。</div>", unsafe_allow_html=True)

            for i, hit in enumerate(filtered_hits[:st.session_state.visible_count]):
                row = df.iloc[hit["corpus_id"]]
                with st.container():
                    st.markdown(f'<div class="qa-box"><strong>{row["question"]}</strong>', unsafe_allow_html=True)
                    with st.expander("▼ 回答を見る", expanded=False, key=make_safe_key(user_input, i)):
                        formatted = format_conversation(str(row['answer']))
                        st.markdown(formatted, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.visible_count < num_hits:
                if st.button("🔽 もっと表示する"):
                    st.session_state.visible_count += 10
                    st.rerun()
