import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import html
import re

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

# ヘッダー
st.markdown("""
<div style='background-color: #e3f3ec; padding: 2rem 1rem; border-radius: 6px; text-align: center;'>
    <h1 style='color: #004d66;'>【TAARS】FAQ検索チャット</h1>
    <p style='font-size: 1.1rem;'>過去のFAQから似た質問と回答を検索できます</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")
    col_map = {
        "問い合わせ内容": "question",
        "返信内容": "answer",
        "ジャンル": "genre"
    }
    df = df.rename(columns=col_map)
    df = df[["question", "answer", "genre"]].dropna()
    return df

@st.cache_data
def load_masking_lists():
    df_pm = pd.read_excel("PM担当者一覧.xlsx")
    df_bld = pd.read_excel("物件一覧.xlsx")
    pm_names = set(df_pm.iloc[:, 0].astype(str) + df_pm.iloc[:, 1].astype(str))
    building_names = set(df_bld.iloc[:, 0].astype(str))
    return pm_names, building_names

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = [normalize_text(q) for q in df["question"].tolist()]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return model, embeddings

def normalize_text(text):
    # あいさつ・定型句の除去
    greetings = [
        r"いつもお世話になっております", r"大変お世話になっております", r"お世話になっております", r"お世話になっています",
        r"よろしくお願いいたします", r"よろしくお願いします", r"お願いいたします", r"お願い致します",
        r"ご確認ください", r"ご連絡ください", r"ご対応お願いいたします", r"ご対応お願いします",
        r"失礼いたします", r"失礼します"
    ]
    for phrase in greetings:
        text = re.sub(phrase, "", text)
    return text.strip()

def apply_masking(text, pm_names, building_names):
    for name in pm_names:
        text = text.replace(name, "〇〇さん")
    for name in building_names:
        text = text.replace(name, "〇〇物件")
    return text

def format_conversation(text):
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        content = html.escape(line)
        if "[サポート]" in line:
            body = content.replace("[サポート]", "")
            formatted = f"<div style='background-color:#e6f7ff; padding:8px 12px; border-radius:6px; margin-bottom:6px;'>💬 サポート：{body}</div>"
        elif "[ユーザー]" in line:
            body = content.replace("[ユーザー]", "")
            formatted = f"<div style='background-color:#f0f0f0; padding:8px 12px; border-radius:6px; margin-bottom:6px;'>👤 ユーザー：{body}</div>"
        else:
            formatted = f"<div style='padding:8px 12px; margin-bottom:6px;'>{content}</div>"
        formatted_lines.append(formatted)
    return "\n".join(formatted_lines)

def show_chat_page(df, model, corpus_embeddings, pm_names, building_names):
    st.subheader("質問を入力してください")

    st.markdown("""
    <div style="border:1px solid #ddd; padding: 1rem; border-radius: 6px; background-color: #f9f9f9; margin-bottom: 1rem;">
    <strong>入力例：</strong><br>
    ・契約書を再発行したい<br>
    ・物件の確認方法<br>
    ・担当者に連絡したい
    </div>
    """, unsafe_allow_html=True)

    if "visible_count" not in st.session_state:
        st.session_state.visible_count = 10
    user_input = st.text_input("", "")

    if user_input:
        st.session_state.visible_count = 10
        with st.spinner("検索中..."):
            query_embedding = model.encode(normalize_text(user_input), convert_to_tensor=True)
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
            question = html.escape(apply_masking(row["question"], pm_names, building_names))
            answer = apply_masking(str(row["answer"]), pm_names, building_names)
            st.markdown(f"""
            <div class="qa-container">
                <strong>{question}</strong>
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer;">▼ 回答を見る</summary>
                    <div style="margin-top: 0.5rem;">
                        {format_conversation(answer)}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.visible_count < num_hits:
            if st.button("🔽 もっと表示する"):
                st.session_state.visible_count += 10
                st.rerun()

def show_genre_page(df, pm_names, building_names):
    st.subheader("ジャンルを選んでFAQを表示")
    genres = sorted(df["genre"].dropna().unique())
    selected_genre = st.selectbox("ジャンルを選択", ["-- 選択してください --"] + genres)

    if selected_genre != "-- 選択してください --":
        filtered_df = df[df["genre"] == selected_genre]
        st.write(f"{len(filtered_df)} 件のFAQが見つかりました。")
        for _, row in filtered_df.iterrows():
            question = html.escape(apply_masking(row["question"], pm_names, building_names))
            answer = apply_masking(str(row["answer"]), pm_names, building_names)
            st.markdown(f"""
            <div class="qa-container">
                <strong>{question}</strong>
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer;">▼ 回答を見る</summary>
                    <div style="margin-top: 0.5rem;">
                        {format_conversation(answer)}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

# データとモデル読み込み
df = load_data()
pm_names, building_names = load_masking_lists()
model, corpus_embeddings = load_model_and_embeddings(df)

# ページ切替
page = st.sidebar.radio("表示ページを選んでください", ["類似QA検索チャット", "ジャンル別FAQ一覧"])
if page == "類似QA検索チャット":
    show_chat_page(df, model, corpus_embeddings, pm_names, building_names)
else:
    show_genre_page(df, pm_names, building_names)
