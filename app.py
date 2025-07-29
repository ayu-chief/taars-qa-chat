import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import html
import re

st.set_page_config(page_title="【TAARS】FAQ検索チャット", layout="wide")

# カスタムCSS（背景・カード・囲み）
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

# アプリヘッダー
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

# 入力タイトル
st.markdown("### 質問を入力してください")

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

@st.cache_data
def load_masking_lists():
    # PM担当者一覧（氏名を連結）
    df_pm = pd.read_excel("PM担当者一覧.xlsx")
    pm_names = set(df_pm["姓"].astype(str) + df_pm["名"].astype(str))

    # 物件一覧（物件名列が2列目想定）
    df_building = pd.read_excel("物件一覧.xlsx", header=4)  # 5行目からデータ
    building_names = set(df_building.iloc[:, 1].dropna().astype(str))  # B列（index=1）

    return list(pm_names), list(building_names)

# マスキング関数
def mask_sensitive_info(text, names, buildings):
    for name in names:
        text = re.sub(rf"\b{re.escape(name)}\b", "〇〇さん", text)
    for bld in buildings:
        text = re.sub(rf"\b{re.escape(bld)}\b", "〇〇物件", text)
    return text

# 会話フォーマット（発言者ごとに背景色）＋マスキング
def format_conversation(text, names, buildings):
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        masked = mask_sensitive_info(line, names, buildings)
        content = html.escape(masked)
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

# データ・モデル・マスキングリスト読込
df = load_data()
model, corpus_embeddings = load_model_and_embeddings(df)
pm_names, building_names = load_masking_lists()

# ユーザー入力
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
            question = row["question"]
            answer = row["answer"]

            st.markdown(f"""
            <div class="qa-container">
                <strong>{html.escape(question)}</strong>
                <details style="margin-top: 0.5rem;">
                    <summary style="cursor: pointer;">▼ 回答を見る</summary>
                    <div style="margin-top: 0.5rem;">
                        {format_conversation(str(answer), pm_names, building_names)}
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
