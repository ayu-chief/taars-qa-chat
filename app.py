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

# 入力例
st.markdown("""
**入力例：**  
- ログインできない  
- 支払い方法を教えてください  
- 契約申請について  
""")

# タイトル
st.markdown("### 質問を入力してください")

if "visible_count" not in st.session_state:
    st.session_state.visible_count = 10

# 定型文除去関数
def clean_text(text):
    patterns = [
        r"(いつも)?(大変)?お世話になって(おり)?ます",
        r"お手数(を)?おかけします(が)?",
        r"恐れ入ります(が)?",
        r"(何卒)?宜しく(お願い)?(いた)?します",
        r"(どうぞ)?よろしく(おねがい)?します",
        r"失礼(いた)?します",
        r"ご確認(の)?ほど(、)?(よろしく)?(お願いいたします)?",
        r"以上、?よろしく(お願いいたします)?",
        r"ありがとうございます。",
        r"ありがとう(ございます)?。",
        r"ご教示(ください)?",
        r"ご回答(いただけ)?(ます)?と幸いです",
        r"ご対応(を)?(お願いいたします)?",
        r"ご連絡(ください)?(ます)?ようお願いいたします",
        r"ご連絡(お)?待ちしております",
        r"ご迷惑(を)?おかけします(が)?",
        r"ご案内(ください)?",
        r"お知らせ(いた)?します",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 文頭の句読点・記号を削除
    text = re.sub(r"^[、。・\s　]+", "", text)
    return text.strip()

@st.cache_data
def load_data():
    df = pd.read_csv("qa_data_with_genre.csv", encoding="utf-8")
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)
    df = df.rename(columns={
        "問い合わせ内容": "question",
        "返信内容": "answer",
        "ジャンル": "genre"
    })
    df = df[["question", "answer", "genre"]].dropna()
    # 不要な定型文削除
    df["question"] = df["question"].astype(str).map(clean_text)
    df["answer"] = df["answer"].astype(str).map(clean_text)
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
    embeddings = model.encode(df["question"].tolist(), convert_to_tensor=True)
    return model, embeddings

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

# データ読み込み
df = load_data()
pm_names, building_names = load_masking_lists()
model, corpus_embeddings = load_model_and_embeddings(df)

# 入力
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
else:
    st.session_state.visible_count = 10


# -------------------------------
# ページ2：ジャンル別FAQ一覧
# -------------------------------
else:
    st.title("ジャンル別FAQ一覧")
    genre_list = sorted(df["ジャンル"].dropna().unique())
    selected_genre = st.selectbox("ジャンルを選択してください", ["すべて"] + genre_list)

    if selected_genre == "すべて":
        filtered_df = df
    else:
        filtered_df = df[df["ジャンル"] == selected_genre]

    if filtered_df.empty:
        st.warning("該当するFAQが見つかりませんでした。")
    else:
        for idx, row in filtered_df.iterrows():
            question = html.escape(apply_masking(row["質問"], pm_names, building_names))
            answer = apply_masking(str(row["回答"]), pm_names, building_names)
            st.markdown(f"""
            <div style='background-color: #ffffff; border-left: 5px solid #e3f3ec; padding: 1rem; margin-bottom: 1.5rem; border-radius: 8px; box-shadow: 0 0 4px rgba(0,0,0,0.05);'>
                <strong>{question}</strong>
                <details style='margin-top: 0.5rem;'>
                    <summary style='cursor: pointer;'>▼ 回答を見る</summary>
                    <div style='margin-top: 0.5rem;'>
                        {format_conversation(answer)}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)
