import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import html

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")

# Tayori風スタイル（背景色やカード）
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
.st-expanderHeader {
    background-color: #e0f7fa !important;
}
.qa-container {
    background-color: #ffffff;
    border-left: 5px solid #e3f3ec;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 0 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ヘッダー（中央揃え＋背景色）
st.markdown("""
<div style='background-color: #e3f3ec; padding: 2rem 1rem; border-radius: 6px; text-align: center;'>
    <h1 style='color: #004d66;'>【TAARS】FAQ検索チャット</h1>
    <p style='font-size: 1.1rem;'>過去のFAQから似た質問と回答を検索できます</p>
</div>
""", unsafe_allow_html=True)

# 入力例
st.markdown("""
**入力例：**  
- ログインできない  
- 支払い方法を教えてください  
- 契約申請について  
""")

# タイトル
st.markdown("### 質問を入力してください")

# 初期表示数のセッション管理
if "visible_count" not in st.session_state:
    st.session_state.visible_count = 10

# データ読み込み
@st.cache_data
def load_data():
    return pd.read_csv("qa_data.csv", encoding="utf-8")

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

def format_conversation(text):
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
            st.success(f"{num_hits} 件の結果が見つかりました。")
            if num_hits > 10:
                st.info("結果が多いため、質問をさらに具体的にすると絞り込みやすくなります。")

        # 区切り線
        st.markdown("""<div style='background-color: #e3f3ec; height: 2px; margin: 2rem 0;'></div>""", unsafe_allow_html=True)

        # 説明文（余白なし）
        st.markdown("""<div style='background-color: #d6e8f3; padding: 0.5rem 1rem; font-size: 0.9rem;'>
            💬 はサポート、👤 はユーザーの発言を表しています。
        </div>""", unsafe_allow_html=True)

        # QAカード表示
        for hit in filtered_hits[:st.session_state.visible_count]:
            row = df.iloc[hit["corpus_id"]]
            question = row["question"]
            answer = row["answer"]

            st.markdown(f"""
            <div class="qa-container">
                <strong>{html.escape(question)}</strong>
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer;">▶ 回答を見る</summary>
                    <div style="margin-top: 0.5rem;">
                        {html.escape(format_conversation(str(answer))).replace("\\n", "<br>").replace("\n", "<br>")}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

        # もっと見るボタン
        if st.session_state.visible_count < num_hits:
            if st.button("🔽 もっと表示する"):
                st.session_state.visible_count += 10
                st.rerun()
else:
    st.session_state.visible_count = 10
