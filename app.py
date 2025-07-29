import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import html

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")

# データ読み込み
@st.cache_data
def load_data():
    return pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")

# モデルと埋め込み読み込み
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["質問"].tolist(), convert_to_tensor=True)
    return model, embeddings

# 会話フォーマット
def format_conversation(text):
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        content = html.escape(line)
        if "[サポート]" in line:
            body = content.replace("[サポート]", "")
            formatted = f"<div style='background-color:#e6f7ff; padding:8px 12px; border-radius:6px;'>💬 サポート：{body}</div>"
        elif "[ユーザー]" in line:
            body = content.replace("[ユーザー]", "")
            formatted = f"<div style='background-color:#f0f0f0; padding:8px 12px; border-radius:6px;'>👤 ユーザー：{body}</div>"
        else:
            formatted = f"<div style='padding:8px 12px;'>{content}</div>"
        formatted_lines.append(formatted)
    return "\n".join(formatted_lines)

# ページ選択
st.sidebar.title("ページ切替")
page = st.sidebar.radio("表示するページを選んでください", ("チャット検索", "ジャンル別FAQ"))

# データロード
df = load_data()

if page == "チャット検索":
    st.markdown("<h1 style='text-align:center;'>【TAARS】FAQ検索チャット</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>過去のFAQから似た質問と回答を検索できます</p>", unsafe_allow_html=True)

    st.markdown("**入力例：**<br>- 契約書を再発行したい<br>- 物件の確認方法<br>- 担当者に連絡したい", unsafe_allow_html=True)
    st.markdown("### 質問を入力してください")

    if "visible_count" not in st.session_state:
        st.session_state.visible_count = 10

    model, corpus_embeddings = load_model_and_embeddings(df)
    user_input = st.text_input("")

    if user_input:
        with st.spinner("検索中..."):
            query_embedding = model.encode(user_input, convert_to_tensor=True)
            results = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(df))[0]
            filtered_hits = [hit for hit in results if hit["score"] >= 0.5]
            num_hits = len(filtered_hits)

            if num_hits == 0:
                st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
            else:
                st.markdown(f"<p style='color: #0a7f4d; font-weight: 500;'>{num_hits} 件の結果が見つかりました。</p>", unsafe_allow_html=True)
                if num_hits > 10:
                    st.markdown("<p style='color: #1565c0;'>結果が多いため、質問を <strong>簡潔に</strong> すると絞り込みやすくなります。</p>", unsafe_allow_html=True)

                for hit in filtered_hits[:st.session_state.visible_count]:
                    row = df.iloc[hit["corpus_id"]]
                    question = row["質問"]
                    answer = row["回答"]

                    st.markdown(f"""
                    <div style="background-color: #ffffff; border-left: 5px solid #e3f3ec; padding: 1rem; margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 0 4px rgba(0,0,0,0.05);">
                        <strong>{html.escape(question)}</strong>
                        <details style="margin-top: 0.5rem;">
                            <summary style="cursor: pointer;">▼ 回答を見る</summary>
                            <div style="margin-top: 0.5rem;">{format_conversation(str(answer))}</div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)

                if st.session_state.visible_count < num_hits:
                    if st.button("🔽 もっと表示する"):
                        st.session_state.visible_count += 10
                        st.rerun()
    else:
        st.session_state.visible_count = 10

elif page == "ジャンル別FAQ":
    st.markdown("<h1>ジャンル別 よくある質問集</h1>", unsafe_allow_html=True)

    genre_options = df["ジャンル"].dropna().unique().tolist()
    selected_genre = st.sidebar.selectbox("ジャンルを選択", genre_options)

    filtered = df[df["ジャンル"] == selected_genre]

    st.markdown(f"### ジャンル：「{selected_genre}」 に関するFAQ（{len(filtered)}件）")

    for _, row in filtered.iterrows():
        st.markdown(f"""
        <div style="background-color: #ffffff; border-left: 5px solid #e3f3ec; padding: 1rem; margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 0 4px rgba(0,0,0,0.05);">
            <strong>{html.escape(row['質問'])}</strong>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer;">▼ 回答を見る</summary>
                <div style="margin-top: 0.5rem;">{format_conversation(str(row['回答']))}</div>
            </details>
        </div>
        """, unsafe_allow_html=True)
