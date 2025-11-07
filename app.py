import os
from typing import Literal

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# .env読み込み（ローカル用）
load_dotenv()

# Secrets（Streamlit Cloud用）も考慮
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        OPENAI_API_KEY = None

def call_llm(user_text: str, expert: Literal["マーケ戦略家", "ソフトウェア設計者"]) -> str:
    system_messages = {
        "マーケ戦略家": (
            "あなたは厳密なデータドリブン思考のマーケティング戦略家です。"
            "市場分析、ペルソナ、4P/3C、ファネル、CAC/LTVを踏まえ、"
            "実行可能な打ち手を見出し、根拠を簡潔に提示してください。"
            "日本語で、具体例と簡単なチェックリストも添えて答えてください。"
        ),
        "ソフトウェア設計者": (
            "あなたは堅牢で拡張可能な設計を重視するソフトウェアアーキテクトです。"
            "要件の分解、非機能要件、アーキテクチャ選定、データ設計、"
            "トレードオフを明示しながら、日本語でステップごとに提案してください。"
        ),
    }

    if not OPENAI_API_KEY:
        return (
            "【エラー】OpenAI APIキーが見つかりませんでした。\n"
            "ローカルは .env に OPENAI_API_KEY=xxxxx を設定、\n"
            "Cloudは Settings→Secrets で登録してください。"
        )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_role}"),
            ("human", "ユーザー入力:\n{user_text}\n\n上記に日本語で回答してください。"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {"system_role": system_messages[expert], "user_text": user_text.strip()}
    )

st.set_page_config(page_title="Streamlit LLM App (LangChain)", page_icon="💬", layout="centered")
st.title("💬 Streamlit × LangChain × OpenAI : LLMアプリ")

with st.expander("ℹ️ このアプリの概要と使い方", expanded=True):
    st.markdown(
        """
**概要**  
- 入力テキストをLangChain経由でOpenAIに渡し、回答を表示します。  
- ラジオボタンの「専門家ロール」でシステムメッセージが切り替わります。

**使い方**  
1. 専門家ロールを選択  
2. テキストを入力  
3. 「送信」ボタンを押す  
        """
    )

expert = st.radio(
    "専門家ロールを選んでください：",
    options=("マーケ戦略家", "ソフトウェア設計者"),
    horizontal=True,
)

with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area(
        "入力テキスト",
        placeholder="例）ECサイトの新規顧客獲得を月100人に増やしたい。現状はSNS流入のみです。",
        height=150,
    )
    submitted = st.form_submit_button("送信")

if submitted:
    if not user_input.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLMが考えています…"):
            answer = call_llm(user_input, expert)
        st.subheader("回答")
        st.write(answer)

st.caption("※ APIキーはGitHubにコミットしないでください（.env／Secretsを使用）。")
