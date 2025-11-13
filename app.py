import os
from typing import Literal

import streamlit as st
from streamlit.components.v1 import html as st_html

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

st.set_page_config(page_title="Streamlit LLM & 株式会社Ring", layout="wide")

PAGE_OPTIONS = {
    "LLMアプリ": "💬 Streamlit × LangChain × OpenAI : LLMアプリ",
    "株式会社Ring 採用情報": "株式会社Ring｜家具家電配送・不用品回収・遺品整理・引越しドライバー求人",
}

page = st.sidebar.selectbox("表示するページを選んでください", options=list(PAGE_OPTIONS.keys()))
st.title(PAGE_OPTIONS[page])

if page == "LLMアプリ":
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
    user_input = st.text_area(
        "入力テキスト",
        placeholder="例）新サービスの売上を月100万円にしたい。どう戦略を立てる？",
    )

    if st.button("送信"):
        if not user_input:
            st.warning("⚠ テキストを入力してください！")
        else:
            answer = call_llm(user_input, expert)
            st.success("✅ 回答：")
            st.write(answer)
else:
    st_html(
        """<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"UTF-8\">
  <title>株式会社Ring｜家具家電配送・不用品回収・遺品整理・引越しドライバー求人</title>
  <meta name=\"description\" content=\"株式会社Ring（リング）は、家具家電配送・不用品回収・遺品整理・引越しを行う成長企業です。完全成果型・安定案件多数のドライバー求人を募集中。未経験からでも高収入を目指せます。\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">

  <!-- SEO用キーワード（必要に応じて調整） -->
  <meta name=\"keywords\" content=\"株式会社Ring,リング,家具家電配送,不用品回収,遺品整理,引越し,ドライバー求人,配送ドライバー,軽貨物,求人\">

  <!-- 構造化データ（Organization + JobPostingのベース） -->
  <script type=\"application/ld+json\">
  {
    \"@context\": \"https://schema.org\",
    \"@type\": \"Organization\",
    \"name\": \"株式会社Ring\",
    \"alternateName\": \"リング\",
    \"url\": \"https://example.com/\",
    \"telephone\": \"08055306427\",
    \"foundingDate\": \"2024-10\",
    \"founder\": {
      \"@type\": \"Person\",
      \"name\": \"中條 瞭太\"
    }
  }
  </script>

  <style>
    :root {
      --blue: #005bbb;
      --blue-light: #e6f2ff;
      --yellow: #ffd400;
      --white: #ffffff;
      --text-main: #222222;
      --text-sub: #555555;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Helvetica Neue\", Arial, \"游ゴシック体\", \"YuGothic\", \"メイリオ\", sans-serif;
      color: var(--text-main);
      background-color: #f5f7fb;
      line-height: 1.7;
    }

    a {
      text-decoration: none;
      color: inherit;
    }

    img {
      max-width: 100%;
      display: block;
    }

    header {
      background-color: var(--white);
      border-bottom: 1px solid #dde3ee;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .header-inner {
      max-width: 1080px;
      margin: 0 auto;
      padding: 8px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .logo {
      font-weight: 700;
      font-size: 1.1rem;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--blue);
    }

    .logo-mark {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: radial-gradient(circle at 30% 30%, var(--yellow), var(--blue));
    }

    nav {
      display: flex;
      gap: 16px;
      font-size: 0.9rem;
    }

    nav a {
      padding: 4px 8px;
      border-radius: 999px;
      transition: background-color 0.2s ease;
    }

    nav a:hover {
      background-color: var(--blue-light);
    }

    .header-cta {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.9rem;
    }

    .tel {
      font-weight: 700;
      color: var(--blue);
    }

    .btn-primary {
      background-color: var(--yellow);
      color: #333;
      border-radius: 999px;
      padding: 8px 16px;
      font-weight: 700;
      font-size: 0.9rem;
      border: 2px solid var(--yellow);
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 10px rgba(0,0,0,0.12);
      background-color: #ffe766;
    }

    .btn-outline {
      background-color: transparent;
      color: var(--blue);
      border-radius: 999px;
      padding: 8px 16px;
      font-weight: 600;
      font-size: 0.9rem;
      border: 1px solid var(--blue);
      cursor: pointer;
      transition: background-color 0.15s ease, color 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .btn-outline:hover {
      background-color: var(--blue);
      color: var(--white);
    }

    main {
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px 16px 80px;
    }

    /* HERO */
    .hero {
      margin-top: 16px;
      background: linear-gradient(135deg, var(--blue) 0%, #0a6fd6 50%, #0d8ce8 100%);
      border-radius: 20px;
      padding: 24px 20px;
      color: var(--white);
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
      gap: 20px;
    }

    .hero-copy h1 {
      font-size: 1.9rem;
      margin-bottom: 12px;
    }

    .hero-copy h1 span {
      background: linear-gradient(90deg, #ffd400, #fff9b0);
      -webkit-background-clip: text;
      color: transparent;
    }

    .hero-copy p {
      font-size: 0.95rem;
      margin-bottom: 16px;
      max-width: 32em;
    }

    .hero-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }

    .badge {
      background-color: rgba(255,255,255,0.12);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      border: 1px solid rgba(255,255,255,0.3);
      white-space: nowrap;
    }

    .hero-cta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }

    .hero-note {
      font-size: 0.8rem;
      opacity: 0.9;
    }

    .hero-visual {
      background-color: rgba(255,255,255,0.07);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid rgba(255,255,255,0.25);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 12px;
    }

    .hero-image-placeholder {
      background-color: rgba(255,255,255,0.15);
      border-radius: 12px;
      height: 140px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.9rem;
      font-weight: 600;
      border: 1px dashed rgba(255,255,255,0.7);
    }

    .hero-meta {
      font-size: 0.8rem;
      opacity: 0.9;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 6px;
    }

    /* SECTION 共通 */
    section {
      margin-top: 40px;
    }

    .section-label {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--blue);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }

    .section-title {
      font-size: 1.4rem;
      margin-bottom: 8px;
    }

    .section-lead {
      font-size: 0.9rem;
      color: var(--text-sub);
      margin-bottom: 20px;
      max-width: 40em;
    }

    /* 強み */
    .cards-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }

    .card {
      background-color: var(--white);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid #dde3ee;
      box-shadow: 0 4px 8px rgba(15, 40, 80, 0.03);
    }

    .card-title {
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 8px;
      color: var(--blue);
    }

    .card-badge {
      display: inline-block;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 999px;
      background-color: var(--blue-light);
      color: var(--blue);
      margin-bottom: 6px;
    }

    .card p {
      font-size: 0.85rem;
      color: var(--text-sub);
    }

    /* 募集要項 */
    .job-layout {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      align-items: flex-start;
    }

    .job-table {
      width: 100%;
      border-collapse: collapse;
      background-color: var(--white);
      border-radius: 12px;
      overflow: hidden;
      font-size: 0.85rem;
    }

    .job-table th,
    .job-table td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf0f7;
      vertical-align: top;
    }

    .job-table th {
      width: 30%;
      background-color: #f5f7fb;
      font-weight: 600;
      color: var(--text-main);
    }

    .job-table tr:last-child th,
    .job-table tr:last-child td {
      border-bottom: none;
    }

    .job-highlight {
      background-color: #fff9d6;
      border-radius: 10px;
      padding: 12px;
      font-size: 0.85rem;
      margin-bottom: 10px;
    }

    .job-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 6px;
    }

    .job-tag {
      font-size: 0.75rem;
      padding: 3px 10px;
      border-radius: 999px;
      border: 1px solid var(--blue);
      color: var(--blue);
      background-color: var(--white);
    }

    .job-image-placeholder {
      background-color: var(--white);
      border-radius: 12px;
      padding: 12px;
      border: 1px dashed #ccd6ea;
      text-align: center;
      font-size: 0.85rem;
      height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-sub);
    }

    /* 1日の流れ */
    .timeline {
      background-color: var(--white);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid #dde3ee;
    }

    .timeline-item {
      display: grid;
      grid-template-columns: 80px minmax(0, 1fr);
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px dashed #e1e6f2;
    }

    .timeline-item:last-child {
      border-bottom: none;
    }

    .timeline-time {
      font-weight: 700;
      color: var(--blue);
      font-size: 0.85rem;
    }

    .timeline-content-title {
      font-size: 0.92rem;
      font-weight: 600;
      margin-bottom: 2px;
    }

    .timeline-content-text {
      font-size: 0.8rem;
      color: var(--text-sub);
    }

    /* よくある質問 */
    .faq-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .faq-item {
      background-color: var(--white);
      border-radius: 12px;
      padding: 12px;
      border: 1px solid #dde3ee;
      font-size: 0.85rem;
    }

    .faq-q {
      font-weight: 700;
      color: var(--blue);
      margin-bottom: 4px;
    }

    .faq-a {
      color: var(--text-sub);
    }

    /* 会社情報 */
    .company-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
      gap: 20px;
      align-items: flex-start;
    }

    .company-table {
      width: 100%;
      border-collapse: collapse;
      background-color: var(--white);
      border-radius: 12px;
      overflow: hidden;
      font-size: 0.85rem;
    }

    .company-table th,
    .company-table td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf0f7;
      vertical-align: top;
    }

    .company-table th {
      width: 30%;
      background-color: #f5f7fb;
      font-weight: 600;
    }

    .company-table tr:last-child th,
    .company-table tr:last-child td {
      border-bottom: none;
    }

    .company-image-placeholder {
      background-color: var(--white);
      border-radius: 12px;
      padding: 12px;
      border: 1px dashed #ccd6ea;
      text-align: center;
      font-size: 0.85rem;
      height: 150px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-sub);
    }

    /* 応募フォーム（ダミー） */
    .contact {
      background-color: var(--white);
      border-radius: 16px;
      padding: 16px;
      border: 1px solid #dde3ee;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px 16px;
    }

    .form-field {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 0.85rem;
    }

    .form-field.full {
      grid-column: 1 / -1;
    }

    .form-field label {
      font-weight: 600;
    }

    .form-field span {
      font-size: 0.75rem;
      color: var(--text-sub);
    }

    input[type=\"text\"],
    input[type=\"tel\"],
    input[type=\"email\"],
    textarea {
      border-radius: 8px;
      border: 1px solid #ccd6ea;
      padding: 8px 10px;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    textarea {
      resize: vertical;
      min-height: 120px;
    }

    input:focus,
    textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 2px rgba(0, 91, 187, 0.15);
    }

    .form-note {
      margin-top: 8px;
      font-size: 0.78rem;
      color: var(--text-sub);
    }

    .form-actions {
      margin-top: 12px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    footer {
      background-color: #0c1f3d;
      color: var(--white);
      padding: 20px 16px;
      margin-top: 40px;
    }

    .footer-inner {
      max-width: 1080px;
      margin: 0 auto;
      font-size: 0.78rem;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      justify-content: space-between;
      align-items: center;
    }

    .footer-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .footer-links a {
      opacity: 0.9;
    }

    .footer-links a:hover {
      opacity: 1;
      text-decoration: underline;
    }

    /* レスポンシブ */
    @media (max-width: 840px) {
      .hero {
        grid-template-columns: minmax(0, 1fr);
      }

      nav {
        display: none;
      }

      .cards-3 {
        grid-template-columns: minmax(0, 1fr);
      }

      .job-layout {
        grid-template-columns: minmax(0, 1fr);
      }

      .faq-list {
        grid-template-columns: minmax(0, 1fr);
      }

      .company-layout {
        grid-template-columns: minmax(0, 1fr);
      }

      .form-grid {
        grid-template-columns: minmax(0, 1fr);
      }

      header {
        position: static;
      }

      main {
        padding-top: 16px;
      }
    }

    @media (max-width: 480px) {
      .hero-copy h1 {
        font-size: 1.5rem;
      }
    }
  </style>
</head>
<body>

<header>
  <div class=\"header-inner\">
    <div class=\"logo\">
      <div class=\"logo-mark\"></div>
      <div>
        <div>株式会社Ring</div>
        <div style=\"font-size:0.7rem; color:#666;\">家具家電配送・不用品回収・遺品整理・引越し</div>
      </div>
    </div>
    <nav>
      <a href=\"#strength\">選ばれる理由</a>
      <a href=\"#job\">募集要項</a>
      <a href=\"#flow\">1日の流れ</a>
      <a href=\"#faq\">よくある質問</a>
      <a href=\"#company\">会社概要</a>
    </nav>
    <div class=\"header-cta\">
      <div class=\"tel\">TEL：080-5530-6427</div>
      <a href=\"#contact\" class=\"btn-primary\">今すぐ応募する</a>
    </div>
  </div>
</header>

<main>
  <!-- HERO -->
  <section class=\"hero\">
    <div class=\"hero-copy\">
      <h1>
        <span>配送のプロ</span>として、<br>
        <span>安定して稼げるドライバー</span>へ。
      </h1>
      <div class=\"hero-badges\">
        <div class=\"badge\">未経験歓迎</div>
        <div class=\"badge\">家具家電配送・不用品回収</div>
        <div class=\"badge\">2024年10月設立の新しい会社</div>
        <div class=\"badge\">しっかり稼ぎたい方、大歓迎</div>
      </div>
      <p>
        株式会社Ring（リング）は、家具家電配送・不用品回収・遺品整理・引越しを手がける成長企業です。
        一緒に会社をつくっていく仲間として、<strong>長く安心して働けるドライバー</strong>を募集しています。
      </p>
      <div class=\"hero-cta-row\">
        <a href=\"#contact\" class=\"btn-primary\">応募フォームへ進む</a>
        <a href=\"#job\" class=\"btn-outline\">募集要項を見る</a>
        <div class=\"hero-note\">
          まずは相談だけでもOKです。<br>
          「話を聞いてみたい」からお気軽にどうぞ。
        </div>
      </div>
    </div>
    <div class=\"hero-visual\">
      <div class=\"hero-image-placeholder\">
        トップ画像（サンプル）
      </div>
      <div class=\"hero-meta\">
        <div>事業内容：家具家電配送／不用品回収／遺品整理／引越し</div>
        <div>代表取締役：中條 瞭太</div>
        <div>設立：2024年10月</div>
        <div>募集職種：配送ドライバー</div>
      </div>
    </div>
  </section>

  <!-- 選ばれる理由 -->
  <section id=\"strength\">
    <div class=\"section-label\">ADVANTAGE</div>
    <h2 class=\"section-title\">株式会社Ringで働く3つの魅力</h2>
    <p class=\"section-lead\">
      「稼げるか」「続けられるか」「将来が見えるか」。ドライバーが気になるポイントを押さえた環境づくりを大切にしています。
    </p>

    <div class=\"cards-3\">
      <div class=\"card\">
        <div class=\"card-badge\">POINT 01</div>
        <div class=\"card-title\">家具家電配送で安定した案件量</div>
        <p>
          家具家電配送を中心に、年間を通じて安定した案件があります。繁忙期だけでなく、<strong>月を通して安定して仕事がある</strong>ので、計画的に収入を組み立てやすいのが特徴です。
        </p>
      </div>
      <div class=\"card\">
        <div class=\"card-badge\">POINT 02</div>
        <div class=\"card-title\">不用品回収・遺品整理で単価の高い案件も</div>
        <p>
          不用品回収や遺品整理、引越しなど、<strong>付加価値の高いサービス</strong>も行っているため、案件次第では売上をしっかり伸ばすことも可能。頑張りが収入に直結します。
        </p>
      </div>
      <div class=\"card\">
        <div class=\"card-badge\">POINT 03</div>
        <div class=\"card-title\">設立まもない会社でコアメンバーになれる</div>
        <p>
          2024年10月設立というスタートアップフェーズの会社です。<strong>これから一緒に会社を大きくしていきたい方</strong>にはぴったりの環境。現場の声が経営に届きやすいのも魅力です。
        </p>
      </div>
    </div>
  </section>

  <!-- 募集要項 -->
  <section id=\"job\">
    <div class=\"section-label\">RECRUIT</div>
    <h2 class=\"section-title\">募集要項</h2>
    <p class=\"section-lead\">
      家具家電配送・不用品回収・遺品整理・引越しに関わるドライバーを募集しています。未経験スタートも歓迎です。
    </p>

    <div class=\"job-layout\">
      <div>
        <div class=\"job-highlight\">
          <strong>こんな方を歓迎します：</strong><br>
          ・体を動かす仕事が好きな方<br>
          ・お客様と丁寧に向き合える方<br>
          ・安定してしっかり稼ぎたい方<br>
          ・新しい会社でコアメンバーとして活躍したい方
        </div>

        <table class=\"job-table\">
          <tr>
            <th>募集職種</th>
            <td>家具家電配送ドライバー／不用品回収・遺品整理・引越しドライバー</td>
          </tr>
          <tr>
            <th>業務内容</th>
            <td>
              ・家庭への家具家電の配送および設置補助<br>
              ・不用品回収作業<br>
              ・遺品整理の現場作業サポート<br>
              ・引越し作業の搬入・搬出 など
            </td>
          </tr>
          <tr>
            <th>雇用形態</th>
            <td>応相談（業務委託／アルバイト／正社員候補 など）</td>
          </tr>
          <tr>
            <th>応募資格</th>
            <td>
              ・普通自動車免許（AT限定可）<br>
              ・学歴・経験不問／未経験歓迎<br>
              ・経験者、家具家電配送経験のある方は優遇
            </td>
          </tr>
          <tr>
            <th>給与・報酬</th>
            <td>
              ・案件や働き方により決定（面談時にご説明します）<br>
              ・安定してしっかり稼げる環境を整えています
            </td>
          </tr>
          <tr>
            <th>勤務時間</th>
            <td>シフト制／案件により変動あり（希望は可能な限り考慮します）</td>
          </tr>
          <tr>
            <th>休日・休暇</th>
            <td>ご希望を伺いながら決定（週〇日休みなど応相談）</td>
          </tr>
          <tr>
            <th>勤務地</th>
            <td>千葉エリアを中心とした各現場（詳細はお問い合わせください）</td>
          </tr>
          <tr>
            <th>募集エリア</th>
            <td>千葉県内および近郊</td>
          </tr>
          <tr>
            <th>選考フロー</th>
            <td>
              応募フォーム送信<br>
              → 担当よりご連絡<br>
              → 面談（オンライン・対面どちらも可）<br>
              → 合否のご連絡
            </td>
          </tr>
        </table>

        <div class=\"job-tags\">
          <div class=\"job-tag\">未経験OK</div>
          <div class=\"job-tag\">学歴不問</div>
          <div class=\"job-tag\">安定案件多数</div>
          <div class=\"job-tag\">新会社の立ち上げメンバー</div>
        </div>
      </div>

      <div>
        <div class=\"job-image-placeholder\">
          現場・トラック写真（サンプル）
        </div>
        <p style=\"font-size:0.8rem; color:var(--text-sub); margin-top:8px;\">
          実際の現場やトラックの写真を掲載することで、働くイメージがより伝わりやすくなります。<br>
          ※後ほど画像に差し替えてください。
        </p>
      </div>
    </div>
  </section>

  <!-- 1日の流れ -->
  <section id=\"flow\">
    <div class=\"section-label\">WORK STYLE</div>
    <h2 class=\"section-title\">1日の仕事の流れ（例）</h2>
    <p class=\"section-lead\">
      家具家電配送ドライバーの、標準的な1日のイメージです。案件や季節によって変動はあります。
    </p>

    <div class=\"timeline\">
      <div class=\"timeline-item\">
        <div class=\"timeline-time\">08:00</div>
        <div>
          <div class=\"timeline-content-title\">出社・当日のルート確認</div>
          <div class=\"timeline-content-text\">
            拠点に集合し、本日の配送ルートや件数を確認。荷物の積み込みも丁寧に行います。
          </div>
        </div>
      </div>
      <div class=\"timeline-item\">
        <div class=\"timeline-time\">09:00</div>
        <div>
          <div class=\"timeline-content-title\">家具家電の配送・設置</div>
          <div class=\"timeline-content-text\">
            個人宅や店舗へ配送。必要に応じて設置や簡単な説明も行います。安全運転と挨拶を大切にします。
          </div>
        </div>
      </div>
      <div class=\"timeline-item\">
        <div class=\"timeline-time\">12:00</div>
        <div>
          <div class=\"timeline-content-title\">休憩</div>
          <div class=\"timeline-content-text\">
            現場の状況に合わせて休憩を取得。しっかり休んで午後の業務に備えます。
          </div>
        </div>
      </div>
      <div class=\"timeline-item\">
        <div class=\"timeline-time\">13:00</div>
        <div>
          <div class=\"timeline-content-title\">不用品回収・遺品整理現場へ</div>
          <div class=\"timeline-content-text\">
            回収先に伺い、不用品の搬出や仕分け、遺品整理のサポートを行います。丁寧な対応が求められる仕事です。
          </div>
        </div>
      </div>
      <div class=\"timeline-item\">
        <div class=\"timeline-time\">17:00</div>
        <div>
          <div class=\"timeline-content-title\">拠点へ戻り、片付け・日報</div>
          <div class=\"timeline-content-text\">
            片付けを行い、簡単な日報を提出。翌日の準備や確認を行って業務終了となります。
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- よくある質問 -->
  <section id=\"faq\">
    <div class=\"section-label\">FAQ</div>
    <h2 class=\"section-title\">よくある質問</h2>
    <p class=\"section-lead\">
      応募前によくいただくご質問をまとめました。その他のご不明点は、お問い合わせフォームからお気軽にどうぞ。
    </p>

    <div class=\"faq-list\">
      <div class=\"faq-item\">
        <div class=\"faq-q\">Q. 未経験でも応募できますか？</div>
        <div class=\"faq-a\">
          はい、未経験の方も歓迎しています。最初は先輩スタッフが同行し、配送の流れやお客様対応などを丁寧にお教えします。
        </div>
      </div>
      <div class=\"faq-item\">
        <div class=\"faq-q\">Q. 車両は自分で用意する必要がありますか？</div>
        <div class=\"faq-a\">
          働き方や契約形態によって異なります。詳細は面談時にご説明しますので、まずはご希望の働き方をお聞かせください。
        </div>
      </div>
      <div class=\"faq-item\">
        <div class=\"faq-q\">Q. どのくらい稼げますか？</div>
        <div class=\"faq-a\">
          案件数や勤務日数によって変わりますが、安定した案件があるため、しっかりと収入を確保しやすい環境です。具体的な目安は面談時にお伝えします。
        </div>
      </div>
      <div class=\"faq-item\">
        <div class=\"faq-q\">Q. 副業やWワークとしても可能ですか？</div>
        <div class=\"faq-a\">
          働き方によってはWワークも相談可能です。ご希望のシフトや稼ぎたい金額を伺いながら決定していきます。
        </div>
      </div>
    </div>
  </section>

  <!-- 会社概要 -->
  <section id=\"company\">
    <div class=\"section-label\">COMPANY</div>
    <h2 class=\"section-title\">会社概要</h2>

    <div class=\"company-layout\">
      <div>
        <table class=\"company-table\">
          <tr>
            <th>社名</th>
            <td>株式会社Ring（リング）</td>
          </tr>
          <tr>
            <th>代表取締役</th>
            <td>中條 瞭太</td>
          </tr>
          <tr>
            <th>設立</th>
            <td>2024年10月</td>
          </tr>
          <tr>
            <th>事業内容</th>
            <td>
              ・家具家電配送<br>
              ・不用品回収<br>
              ・遺品整理<br>
              ・引越し
            </td>
          </tr>
          <tr>
            <th>電話番号</th>
            <td>080-5530-6427</td>
          </tr>
          <tr>
            <th>所在地</th>
            <td>（所在地住所を掲載予定）</td>
          </tr>
          <tr>
            <th>対応エリア</th>
            <td>千葉県内および近郊（詳細はお問い合わせください）</td>
          </tr>
        </table>
      </div>
      <div>
        <div class=\"company-image-placeholder\">
          会社外観・ロゴ等の画像（サンプル）
        </div>
        <p style=\"font-size:0.8rem; color:var(--text-sub); margin-top:8px;\">
          事務所外観やスタッフ集合写真などを掲載すると、安心感・信頼感につながります。<br>
          ※後ほど画像に差し替えてください。
        </p>
      </div>
    </div>
  </section>

  <!-- 応募フォーム -->
  <section id=\"contact\">
    <div class=\"section-label\">ENTRY</div>
    <h2 class=\"section-title\">応募・お問い合わせフォーム</h2>
    <p class=\"section-lead\">
      下記フォームに必要事項をご入力のうえ、送信してください。担当より折り返しご連絡いたします。<br>
      ※実装前のため、現時点ではサンプルフォームです。
    </p>

    <div class=\"contact\">
      <form>
        <div class=\"form-grid\">
          <div class=\"form-field\">
            <label for=\"name\">お名前</label>
            <input type=\"text\" id=\"name\" name=\"name\" placeholder=\"例）山田 太郎\">
          </div>
          <div class=\"form-field\">
            <label for=\"kana\">フリガナ</label>
            <input type=\"text\" id=\"kana\" name=\"kana\" placeholder=\"例）ヤマダ タロウ\">
          </div>
          <div class=\"form-field\">
            <label for=\"tel\">電話番号</label>
            <input type=\"tel\" id=\"tel\" name=\"tel\" placeholder=\"例）08012345678\">
          </div>
          <div class=\"form-field\">
            <label for=\"email\">メールアドレス</label>
            <input type=\"email\" id=\"email\" name=\"email\" placeholder=\"例）example@mail.com\">
          </div>
          <div class=\"form-field full\">
            <label for=\"area\">お住まいのエリア</label>
            <input type=\"text\" id=\"area\" name=\"area\" placeholder=\"例）千葉県〇〇市\">
          </div>
          <div class=\"form-field full\">
            <label for=\"message\">ご質問・ご希望の働き方など</label>
            <textarea id=\"message\" name=\"message\" placeholder=\"ご希望の勤務日数や時間帯、質問事項などをご自由にご記入ください。\"></textarea>
          </div>
        </div>

        <div class=\"form-note\">
          ※送信ボタンはダミーです。実際の運用時には、メール送信機能や応募管理システムとの連携を行ってください。
        </div>

        <div class=\"form-actions\">
          <button type=\"reset\" class=\"btn-outline\">内容をクリア</button>
          <button type=\"submit\" class=\"btn-primary\">この内容で送信する（サンプル）</button>
        </div>
      </form>
    </div>
  </section>
</main>

<footer>
  <div class=\"footer-inner\">
    <div>
      © 株式会社Ring（リング）
    </div>
    <div class=\"footer-links\">
      <span>事業内容：家具家電配送／不用品回収／遺品整理／引越し</span>
      <span>｜</span>
      <a href=\"#job\">採用情報</a>
      <span>｜</span>
      <span>TEL：080-5530-6427</span>
    </div>
  </div>
</footer>

</body>
</html>
        """,
        height=2200,
        scrolling=True,
    )
