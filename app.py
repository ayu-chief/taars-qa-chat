import streamlit as st
import pandas as pd
import re
import html
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

# 不要文言を除去
def clean_text(text):
    patterns = [
        r"(いつも)?大変?お世話になって(おり)?ます",
        r"(何卒)?よろしく(お願い)?いたします",
        r"(何卒)?宜しく(お願い)?(致)?します",
        r"以上、?よろしく(お願い)?いたします",
        r"ご確認のほど(、)?(よろしく)?お願いいたします",
        r"(どうぞ)?よろしく(お願いいたします)?",
        r"失礼いたします。",
        r"ご対応お願いいたします。",
        r"ありがとうございます。",
        r"ご連絡いたします。",
        r"恐れ入りますが",
        r"お忙しいところ(、)?",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()

# マスキング処理
def apply_masking(text, pm_names, building_names):
    for name in sorted(pm_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "〇〇さん", text)
    for name in sorted(building_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "〇〇物件", text)
    return text

# 会話整形
def format_conversation(text):
    lines = text.splitlines()
    formatted = []
    for line in lines:
        content = html.escape(line)
        if "[サポート]" in line:
            body = content.replace("[サポート]", "")
            formatted.append(f"<div style='background-color:#e6f7ff; padding:8px 12px; border-radius:6px; margin-bottom:6px;'>💬 サポート：{body}</div>")
        elif "[ユーザー]" in line:
            body = content.replace("[ユーザー]", "")
            formatted.append(f"<div style='background-color:#f0f0f0; padding:8px 12px; border-radius:6px; margin-bottom:6px;'>👤 ユーザー：{body}</div>")
        else:
            formatted.append(f"<div style='padding:8px 12px; margin-bottom:6px;'>{content}</div>")
    return "\n".join(formatted)

# データ読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)
    df = df.rename(columns={
        "問い合わせ内容": "質問",
        "返信内容": "回答",
        "ジャンル": "ジャンル"
    })
    df = df.dropna(subset=["質問", "回答"])
    df["質問"] = df["質問"].map(clean_text)
    df["回答"] = df["回答"].map(clean_text)
    return df

@st.cache_data
def load_masking_lists():
    df_pm = pd.read_excel("PM担当者一覧.xlsx")
    df_bld = pd.read_excel("物件一覧.xlsx")
    df_pm.columns = df_pm.columns.str.strip()
    df_bld.columns = df_bld.columns.str.strip()
    pm_names = set(df_pm.iloc[:, 0].astype(str) + df_pm.iloc[:, 1].astype(str))
    building_names = set(df_bld.iloc[:, 0].astype(str))
    return pm_names, building_names

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["質問"].tolist(), convert_to_tensor=True)
    return model, embeddings

# 類似検索ページ
def show_chat_page(df, model, embeddings, pm_names, building_names):
    st.subheader("質問を入力してください")
    st.markdown("""
    **入力例：**  
    - ログインできない  
    - 支払い方法を教えてください  
    - 契約申請について  
    """)

    if "visible_count" not in st.session_state:
        st.session_state.visible_count = 10

    query = st.text_input("", "")
    if query:
        with st.spinner("検索中..."):
            query_clean = clean_text(query)
            query_embedding = model.encode(query_clean, convert_to_tensor=True)
            results = util.semantic_search(query_embedding, embeddings, top_k=len(df))[0]
            filtered = [r for r in results if r["score"] >= 0.5]

        if not filtered:
            st.warning("該当するQAが見つかりませんでした。もう少し具体的に入力してください。")
        else:
            st.markdown(f"<p style='color:#0a7f4d; font-weight:500;'>{len(filtered)} 件の結果が見つかりました。</p>", unsafe_allow_html=True)
            if len(filtered) > 10:
                st.markdown("<p style='color:#1565c0;'>結果が多いため、質問を <strong>簡潔に</strong> すると絞り込みやすくなります。</p>", unsafe_allow_html=True)

            st.markdown("<div style='background-color:#e3f3ec; height:2px; margin:2rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='background-color:#d6e8f3; padding:0.5rem 1rem; font-size:0.9rem;'>💬 はサポート、👤 はユーザーの発言を表しています。</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

            for r in filtered[:st.session_state.visible_count]:
                row = df.iloc[r["corpus_id"]]
                masked_q = html.escape(apply_masking(row["質問"], pm_names, building_names))
                masked_a = apply_masking(row["回答"], pm_names, building_names)
                st.markdown(f"""
                <div class="qa-container">
                    <strong>{masked_q}</strong>
                    <details style="margin-top: 0.5rem;">
                        <summary style="cursor: pointer;">▼ 回答を見る</summary>
                        <div style="margin-top: 0.5rem;">
                            {format_conversation(masked_a)}
                        </div>
                    </details>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.visible_count < len(filtered):
                if st.button("🔽 もっと表示する"):
                    st.session_state.visible_count += 10
                    st.rerun()
    else:
        st.session_state.visible_count = 10

# ジャンル別ページ
def show_genre_page(df, pm_names, building_names):
    st.subheader("ジャンル別 FAQ")
    genre_list = sorted(df["ジャンル"].dropna().unique())
    selected = st.selectbox("ジャンルを選択", ["すべて"] + genre_list)

    if selected == "すべて":
        subset = df
    else:
        subset = df[df["ジャンル"] == selected]

    if subset.empty:
        st.info("このジャンルには質問が登録されていません。")
        return

    for _, row in subset.iterrows():
        masked_q = apply_masking(row["質問"], pm_names, building_names)
        masked_a = apply_masking(row["回答"], pm_names, building_names)
        st.markdown(f"""
        <div class="qa-container">
            <strong>{html.escape(masked_q)}</strong>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer;">▼ 回答を見る</summary>
                <div style="margin-top: 0.5rem;">
                    {format_conversation(masked_a)}
                </div>
            </details>
        </div>
        """, unsafe_allow_html=True)

# メイン
def main():
    st.title("【TAARS】FAQ検索チャット")
    st.caption("過去のFAQから似た質問と回答を検索できます")
    page = st.sidebar.radio("表示ページを選んでください", ["類似QA検索チャット", "ジャンル別FAQ一覧"])

    df = load_data()
    pm_names, building_names = load_masking_lists()
    model, embeddings = load_model_and_embeddings(df)

    if page == "類似QA検索チャット":
        show_chat_page(df, model, embeddings, pm_names, building_names)
    else:
        show_genre_page(df, pm_names, building_names)

if __name__ == "__main__":
    main()
