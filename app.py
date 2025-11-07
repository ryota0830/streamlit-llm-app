import os
import streamlit as st
from typing import Literal

# ✅ ローカルでは .env を読み込む（Cloud 上では .env なしでも動作）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass  # Streamlit Cloud では dotenv は使わない

# ✅ OpenAI API キーの取得（ローカル → Cloud の順で確認）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

# ✅ LangChain と OpenAI を準備
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ LLM 呼び出し関数
def call_llm(user_text: str, expert: Literal["マーケ戦略家", "ソフトウェア設計者"]) -> str:
    system_messages = {
        "マーケ戦略家": (
            "あなたは高度なデータドリブン思考のマーケティング戦略家です。"
            "市場分析、ペルソナ、4P/3C、ファネル、CAC/LTVを踏まえ、"
            "具体案を簡潔に提案してください。"
        ),
        "ソフトウェア設計者": (
            "あなたは堅牢で拡張可能な設計を重視するソフトウェア設計者です。"
            "機能仕様、DB設計、API設計、アーキテクチャ構成、技術選定などを含めて説明してください。"
        ),
    }

    if not OPENAI_API_KEY:
        return "❌ OpenAI APIキーが設定されていません。\n" \
               "・ローカル：.env に OPENAI_API_KEY=xxx を設定\n" \
               "・Streamlit Cloud：Secrets に設定"

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_messages[expert]),
            ("user", "{user_input}")
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": user_text})

# =========================
# ✅ Streamlit UI
# =========================
st.title("💬 Streamlit × LangChain × OpenAI : LLMアプリ")

with st.expander("ℹ️ このアプリの概要と使い方", expanded=False):
    st.write("""
    **概要**  
    - 入力テキストをLangChain経由でOpenAIに渡し、回答を表示します。  
    - 専門家ロールで回答内容が変化します。

    **使い方**  
    1. 専門家ロールを選択  
    2. テキストを入力  
    3. 「送信」ボタンを押す  
    """)

expert = st.radio("専門家ロールを選んでください：", ["マーケ戦略家", "ソフトウェア設計者"])
user_input = st.text_area("入力テキスト", placeholder="例）新サービスの売上を月100万円にしたい。どう戦略を立てる？")

if st.button("送信"):
    if not user_input:
        st.warning("⚠ テキストを入力してください！")
    else:
        answer = call_llm(user_input, expert)
        st.success("✅ 回答：")
        st.write(answer)
