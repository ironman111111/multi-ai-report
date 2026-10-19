import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Multi-AI Report", page_icon="🤖", layout="wide")

# ---------- AI providers (all use the same OpenAI-style API) ----------
PROVIDERS = {
    "Groq (Llama)": {
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "Google Gemini": {
        "key_name": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
    "OpenRouter (free)": {
        "key_name": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
DB_FILE = "reports.db"


# ---------- Helpers ----------
def get_saved_key(key_name):
    try:
        value = st.secrets.get(key_name, "")
    except Exception:
        value = ""
    return value or os.getenv(key_name, "")


def ask_one(provider, model, api_key, question):
    """Ask one AI. Never crashes: errors are returned in the result."""
    start = time.time()
    result = {"provider": provider, "model": model, "answer": "", "error": "", "seconds": 0}
    try:
        client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"], timeout=60)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": question}],
            temperature=0.3,
            max_tokens=1200,
        )
        result["answer"] = (resp.choices[0].message.content or "").strip()
        if not result["answer"]:
            result["error"] = "Empty answer"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["seconds"] = round(time.time() - start, 2)
    return result


def build_report(question, results, summary):
    lines = ["# Multi-AI Answer Report",
             f"*Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}*", "",
             "## Question", question, ""]
    if summary:
        lines += ["## Combined Summary", summary, ""]
    lines.append("## Answers by AI")
    for r in results:
        lines.append(f"### {r['provider']} (`{r['model']}`, {r['seconds']}s)")
        lines.append(r["answer"] or f"_Failed: {r['error']}_")
        lines.append("")
    return "\n".join(lines)


def db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TEXT, question TEXT, report_md TEXT)""")
    return conn


def save_report(question, md):
    with db() as c:
        c.execute("INSERT INTO reports (created, question, report_md) VALUES (?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M"), question, md))


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
selected = {}
for name, cfg in PROVIDERS.items():
    with st.sidebar.expander(name):
        typed = st.text_input("API key (optional if saved in secrets)", type="password", key=f"k_{name}")
        model = st.text_input("Model", value=cfg["model"], key=f"m_{name}")
        key = typed.strip() or get_saved_key(cfg["key_name"])
        if st.checkbox("Use this AI", value=bool(key), disabled=not key, key=f"u_{name}") and key:
            selected[name] = (model.strip(), key)

do_summary = st.sidebar.checkbox("Add combined summary", value=True)
summary_by = st.sidebar.selectbox("Summary written by", list(selected) or ["-"])
page = st.sidebar.radio("Page", ["Ask", "History"])

# ---------- Ask page ----------
if page == "Ask":
    st.title("🤖 Multi-AI Answer Report")
    st.write("Ask one question and compare answers from different AIs.")

    with st.form("ask"):
        question = st.text_area("Your question", height=120)
        go = st.form_submit_button("Ask all AIs", type="primary")

    if go:
        question = question.strip()
        if not question:
            st.warning("Please type a question.")
        elif len(question) > 2000:
            st.warning("Question too long (max 2000 characters).")
        elif not selected:
            st.error("No AI enabled. Add at least one API key in the sidebar.")
        else:
            with st.spinner(f"Asking {len(selected)} AI(s)..."):
                with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                    futures = [pool.submit(ask_one, n, m, k, question)
                               for n, (m, k) in selected.items()]
                    results = [f.result() for f in futures]

                summary = ""
                good = [r for r in results if r["answer"]]
                if do_summary and len(good) >= 2 and summary_by in selected:
                    joined = "\n\n".join(f"### {r['provider']}\n{r['answer']}" for r in good)
                    prompt = (f"Question: {question}\n\nAnswers from different AIs:\n\n{joined}\n\n"
                              "Write a short summary: where the AIs agree, where they differ, "
                              "and a best final answer.")
                    m, k = selected[summary_by]
                    s = ask_one(summary_by, m, k, prompt)
                    summary = s["answer"] or f"(Summary failed: {s['error']})"

                md = build_report(question, results, summary)
                save_report(question, md)
                st.session_state["last"] = (results, summary, md)

    if "last" in st.session_state:
        results, summary, md = st.session_state["last"]
        if summary:
            st.subheader("📝 Combined summary")
            st.info(summary)
        st.subheader("Answers")
        tabs = st.tabs([r["provider"] for r in results])
        for tab, r in zip(tabs, results):
            with tab:
                if r["error"] and not r["answer"]:
                    st.error(r["error"])
                else:
                    st.markdown(r["answer"])
                st.caption(f"Model: {r['model']} · {r['seconds']}s")
        st.subheader("Quick comparison")
        st.dataframe(pd.DataFrame([{
            "AI": r["provider"],
            "Status": "OK" if r["answer"] else "Failed",
            "Seconds": r["seconds"],
            "Words": len(r["answer"].split()),
        } for r in results]), hide_index=True)
        st.download_button("⬇️ Download report (.md)", md,
                           file_name="ai_report.md", mime="text/markdown")

# ---------- History page ----------
else:
    st.title("🗂️ History")
    with db() as c:
        rows = c.execute("SELECT id, created, question, report_md FROM reports ORDER BY id DESC").fetchall()
    if not rows:
        st.info("No reports yet.")
    else:
        labels = {f"{r[1]} - {r[2][:60]}": r for r in rows}
        row = labels[st.selectbox("Pick a report", list(labels))]
        st.markdown(row[3])
        st.download_button("⬇️ Download", row[3], file_name=f"report_{row[0]}.md")
        if st.button("🗑️ Delete this report"):
            with db() as c:
                c.execute("DELETE FROM reports WHERE id=?", (row[0],))
            st.rerun()
