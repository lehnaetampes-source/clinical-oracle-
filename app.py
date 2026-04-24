import os
import time
import json
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ARCHIVE_FILE = "archives_oracle.json"

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "k_val" not in st.session_state:
    st.session_state.k_val = 12
if "expert_overlay" not in st.session_state:
    st.session_state.expert_overlay = True
if "show_scores" not in st.session_state:
    st.session_state.show_scores = False
if "enable_judge" not in st.session_state:
    st.session_state.enable_judge = True

def save_to_archive(history):
    archive_data = []
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            archive_data = json.load(f)
    archive_data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "full_chat": history
    })
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, indent=4, ensure_ascii=False)

st.set_page_config(page_title="THE CLINICAL ORACLE", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 3px solid #0047AB !important; box-shadow: 5px 0 25px rgba(0,71,171,0.6); }
    .oracle-title { font-family: 'Orbitron', sans-serif; color: #0047AB; text-shadow: 0 0 15px #0047AB, 0 0 30px #0000FF; text-align: center; font-size: 3.5rem; font-weight: 900; letter-spacing: 8px; padding: 20px; }
    .nih-subtitle { color: #0047AB; text-align: center; font-family: 'Orbitron'; letter-spacing: 4px; font-size: 0.9rem; margin-top: -20px; margin-bottom: 30px; }
    div[data-baseweb="input"] { border: 2px solid #0047AB !important; background-color: #000000 !important; border-radius: 5px !important; }
    .chat-entry { border-left: 2px solid #0047AB; padding-left: 15px; margin-bottom: 25px; background: rgba(0,71,171,0.05); }
    .judge-box { border: 1px solid #0047AB; padding: 10px; margin-top: 10px; background: rgba(0,71,171,0.08); border-radius: 5px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    .score-high { color: #2DC653; font-weight: bold; }
    .score-mid  { color: #FF9F1C; font-weight: bold; }
    .score-low  { color: #E63946; font-weight: bold; }
    .stMarkdown p, .stMarkdown li, .stMarkdown h3 { color: #FFFFFF !important; }
    .stProgress > div > div > div > div { background-color: #0047AB !important; box-shadow: 0 0 15px #0000FF; }
    .stButton>button { background: #000000 !important; color: #0047AB !important; border: 1px solid #0047AB !important; font-family: 'Orbitron', sans-serif; font-weight: bold; }
    .stButton>button:hover { border: 1px solid #FFFFFF !important; color: #FFFFFF !important; box-shadow: 0 0 15px #0047AB; }
    .stExpander { border: 1px solid #0047AB !important; background: rgba(0,71,171,0.05) !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_oracle():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
    vs  = Chroma(persist_directory="chroma_db", embedding_function=emb)
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.environ.get("MISTRAL_API_KEY")
    )
    return vs, llm, emb

vectorstore, llm, embeddings = load_oracle()

combined_prompt = ChatPromptTemplate.from_template("""
You are a Senior Clinical Data Analyst working with NIH clinical trial protocols.
USER QUESTION: {question}
RETRIEVED CONTEXT:
{context}
INSTRUCTIONS:
1. Answer based ONLY on the provided context.
2. Cite the source file name for every specific claim.
3. If the context mentions scores (VAS, Constant-Murley, DASH, SF-36, etc.), extract exact values.
4. Structure your answer with clear bullet points.
5. If information is not found in the context, explicitly state it.
6. End with a confidence score (0-100%).
ANSWER:
""")

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of RAG systems for clinical research.
Evaluate the following RAG response. Respond ONLY with valid JSON, no explanation outside the JSON.
QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}
Return ONLY this JSON:
{{"faithfulness": <0-10>, "relevance": <0-10>, "completeness": <0-10>, "citation": <0-10>, "feedback": "<one sentence>"}}
""")

def run_rag(question: str, k: int) -> dict:
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    docs      = [r[0] for r in docs_with_scores]
    l2_scores = [r[1] for r in docs_with_scores]
    if not docs:
        return {"answer": "No relevant documents found.", "sources_details": [], "query_used": question, "judge_scores": None}
    query_vec  = np.array(embeddings.embed_query(question)).reshape(1, -1)
    doc_vecs   = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    cos_scores = cosine_similarity(query_vec, doc_vecs)[0]
    context = ""
    sources_info = []
    for i, (doc, cos, l2) in enumerate(zip(docs, cos_scores, l2_scores)):
        src     = Path(doc.metadata.get("source", "Unknown")).name
        quality = "HIGH" if cos > 0.7 else ("MEDIUM" if cos > 0.5 else "LOW")
        context += f"--- DOC {i+1} | {src} | Similarity: {cos:.2%} ({quality}) ---\n{doc.page_content}\n\n"
        sources_info.append({"source": src, "similarity": f"{cos:.2%}", "quality": quality, "content": doc.page_content})
    answer = (combined_prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"answer": answer, "query_used": question, "sources_details": sources_info,
            "context_preview": "\n".join([f"[{s['source']}]: {s['content'][:80]}" for s in sources_info])}

def run_judge(question, context, answer):
    try:
        raw    = (judge_prompt | llm | StrOutputParser()).invoke({"question": question, "context": context, "answer": answer})
        clean  = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        scores["overall"] = round((scores["faithfulness"]+scores["relevance"]+scores["completeness"]+scores["citation"])/4, 1)
        return scores
    except Exception as e:
        return {"faithfulness":0,"relevance":0,"completeness":0,"citation":0,"overall":0,"feedback":f"Eval error: {e}"}

def score_class(v):
    return "score-high" if v >= 7 else ("score-mid" if v >= 4 else "score-low")

# BOOT
if not st.session_state.initialized:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<br><br>", unsafe_allow_html=True)
        _, col, _ = st.columns([1, 2, 1])
        with col:
            if Path("logo.png").exists():
                st.image("logo.png", width=400)
            st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(101):
                time.sleep(0.006)
                bar.progress(i)
    st.session_state.initialized = True
    placeholder.empty()
    st.rerun()

# SIDEBAR
with st.sidebar:
    if Path("logo.png").exists():
        st.image("logo.png", use_column_width=True)  # ✅ FIXED: was use_container_width=True
    st.markdown("<h2 style='color:#0047AB;font-family:Orbitron;text-align:center;'>COMMAND CENTER</h2>", unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.rerun()
    tabs = st.tabs(["SETTINGS", "ARCHIVES"])
    with tabs[0]:
        new_k = st.slider("Scan Depth (Chunks)", 4, 30, st.session_state.k_val)
        if new_k != st.session_state.k_val: st.session_state.k_val = new_k
        new_expert = st.toggle("Expert Data Overlay", value=st.session_state.expert_overlay)
        if new_expert != st.session_state.expert_overlay: st.session_state.expert_overlay = new_expert
        new_scores = st.toggle("Show Similarity Scores", value=st.session_state.show_scores)
        if new_scores != st.session_state.show_scores: st.session_state.show_scores = new_scores
        new_judge = st.toggle("⚖️ LLM-as-Judge", value=st.session_state.enable_judge)
        if new_judge != st.session_state.enable_judge: st.session_state.enable_judge = new_judge
        st.markdown(f"<small style='color:#0047AB'>k actuel : **{st.session_state.k_val}** chunks</small>", unsafe_allow_html=True)
    with tabs[1]:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                history_files = json.load(f)
            for item in reversed(history_files[-5:]):
                if st.button(f"📄 {item['timestamp']}", key=item['timestamp']):
                    st.session_state.chat_history = item['full_chat']
                    st.rerun()
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#0047AB;font-family:Orbitron;font-size:0.7rem;'>MEDICAL AGENT v4.0 ELITE</div>", unsafe_allow_html=True)

# MAIN
st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
st.markdown("<div class='nih-subtitle'>NIH CLINICAL INTELLIGENCE SYSTEM</div>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    st.markdown(f"**>> QUERY:** {entry['query']}")
    st.markdown(f"<div class='chat-entry'>{entry['response']}</div>", unsafe_allow_html=True)
    if entry.get("judge_scores"):
        j = entry["judge_scores"]
        st.markdown(f"""<div class='judge-box'>
        ⚖️ <b>LLM-AS-JUDGE</b> &nbsp;|&nbsp;
        Faithfulness: <span class='{score_class(j["faithfulness"])}'>{j["faithfulness"]}/10</span> &nbsp;|&nbsp;
        Relevance: <span class='{score_class(j["relevance"])}'>{j["relevance"]}/10</span> &nbsp;|&nbsp;
        Completeness: <span class='{score_class(j["completeness"])}'>{j["completeness"]}/10</span> &nbsp;|&nbsp;
        Citation: <span class='{score_class(j["citation"])}'>{j["citation"]}/10</span> &nbsp;|&nbsp;
        <b>Overall: <span class='{score_class(j["overall"])}'>{j["overall"]}/10</span></b><br>
        💬 {j.get("feedback", "")}
        </div>""", unsafe_allow_html=True)

with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input(">> INITIALIZE ORACLE QUERY :")
    submit_button = st.form_submit_button(label='SEND TO CORE')

if submit_button and query:
    with st.spinner("⚡ ORACLE ANALYZING..."):
        result = run_rag(query, k=st.session_state.k_val)
        judge_scores = None
        if st.session_state.enable_judge:
            judge_scores = run_judge(query, result.get("context_preview",""), result["answer"])
        st.session_state.chat_history.append({
            "query": query, "response": result["answer"],
            "query_used": result["query_used"], "sources": result["sources_details"],
            "judge_scores": judge_scores
        })
        st.session_state.last_docs = result["sources_details"]
        st.rerun()

if st.session_state.chat_history:
    st.markdown("---")
    if st.session_state.expert_overlay and st.session_state.last_docs:
        st.markdown("### 📁 RAW DATA CHUNKS (LAST SCAN)")
        for i, src in enumerate(st.session_state.last_docs):
            score_text   = f" | SIM: {src['similarity']}" if st.session_state.show_scores else ""
            quality_icon = "✅" if src["quality"]=="HIGH" else ("⚠️" if src["quality"]=="MEDIUM" else "❌")
            with st.expander(f"{quality_icon} SOURCE {i+1} | {src['source']}{score_text}"):
                st.write(src["content"])
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ARCHIVE FULL SESSION"):
            save_to_archive(st.session_state.chat_history)
            st.success("SESSION PERSISTED.")
    with c2:
        full_text = "\n\n".join([f"Q: {e['query']}\nRewritten: {e.get('query_used','')}\nA: {e['response']}" for e in st.session_state.chat_history])
        st.download_button("📄 DOWNLOAD FULL REPORT", full_text, file_name="full_report.txt")
