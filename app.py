import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import html

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")

# --- ページ選択 ---
page = st.sidebar.radio("ページ選択", ["🔍 質問で探す", "📂 ジャンルで探す"])

# --- データ読込 ---
@st.cache_data
def load_data():
    return pd.read_csv("ジャンル分類付き_TAARSお問い合わせ.csv", encoding="utf-8")

df = load_data()

# --- 会話フォーマット ---
def format_conversation(text):
    lines = str(text).splitlines()
    formatted = []
    for line in lines:
        content = html.escape(line)
        if "[サポート]" in line:
            body = content.replace("[サポート]", "")
            formatted.append(f"<div style='background-color:#e6f7ff; padding:8px 12px; border-radius:6px;'>💬 サポート：{body}</div>")
        elif "[ユーザー]" in line:
            body = content.replace("[ユーザー]", "")
            formatted.append(f"<div style='background-color:#f0f0f0; padding:8px 12px; border-radius:6px;'>👤 ユーザー：{body}</div>")
        else:
            formatted.append(f"<div style='padding:8px 12px;'>{content}</div>")
    return "\n".join(formatted)

# --- 共通のQA表示関数 ---
def display_qa_block(filtered_df):
    for _, row in filtered_df.iterrows():
        question = row["question"]
        answer = row["answer"]
        st.markdown(f"""
        <div class="qa-container">
            <strong>{html.escape(str(question))}</strong>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer;">▼ 回答を見る</summary>
                <div style="margin-top: 0.5rem;">
                    {format_conversation(answer)}
                </div>
            </details>
        </div>
        """, unsafe_allow_html=True)

# --- CSS ---
st.markdown("""
<style>
body { background-color: #f4f8f9; }
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

# --- ページ1: 質問で探す ---
if page == "🔍 質問で探す":
    st.markdown("""
    <div style='background-color: #e3f3ec; padding: 2rem 1rem; border-radius: 6px; text-align: center;'>
        <h1 style='color: #004d66;'>【TAARS】FAQ検索チャット</h1>
        <p style='font-size: 1.1rem;'>過去のFAQから似た質問と回答を検索できます</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**入力例：**  \n- 契約書を再発行したい  \n- 物件の確認方法  \n- 担当者に連絡したい")
    st.markdown("### 質問を入力してください")

    if "visible_count" not in st.session_state:
        st.session_state.visible_count = 10

    @st.cache_resource
    def load_model_and_embeddings(df):
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
        return model, embeddings

    model, corpus_embeddings = load_model_and_embeddings(df)
    user_input = st.text_input("", "")

    if user_input:
        st.session_state.visible_count = 10  # 検索時にリセット
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

                st.markdown("<div style='background-color: #e3f3ec; height: 2px; margin: 2rem 0;'></div>", unsafe_allow_html=True)
                st.markdown("<div style='background-color: #d6e8f3; padding: 0.5rem 1rem; font-size: 0.9rem;'>💬 はサポート、👤 はユーザーの発言を表しています。</div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

                for hit in filtered_hits[:st.session_state.visible_count]:
                    row = df.iloc[hit["corpus_id"]]
                    st.markdown(f"""
                    <div class="qa-container">
                        <strong>{html.escape(str(row['question']))}</strong>
                        <details style="margin-top: 0.5rem;">
                            <summary style="cursor: pointer;">▼ 回答を見る</summary>
                            <div style="margin-top: 0.5rem;">
                                {format_conversation(row['answer'])}
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)

                if st.session_state.visible_count < num_hits:
                    if st.button("🔽 もっと表示する"):
                        st.session_state.visible_count += 10
                        st.rerun()

# --- ページ2: ジャンルで探す ---
elif page == "📂 ジャンルで探す":
    st.markdown("## ジャンルからFAQを探す")
    genre_list = sorted(df["ジャンル"].dropna().unique())
    selected_genre = st.selectbox("ジャンルを選んでください", genre_list)

    genre_df = df[df["ジャンル"] == selected_genre]
    st.markdown(f"**{selected_genre}** に関する {len(genre_df)} 件のQAが見つかりました。")
    display_qa_block(genre_df)
