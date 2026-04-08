"""
rag_utils.py — TRICARE RAG 파이프라인
tricare_core.py 기반으로 팀 컨벤션(make_llm) 적용 버전

사용 예시:
    from rag_utils import make_rag_chain_v3, load_vector_stores
    load_vector_stores()
    answer, docs = make_rag_chain_v3("질문")
"""

import os
import json as _json
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ── 팀 컨벤션: make_llm() ──────────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

def make_llm(temp=0):
    kwargs = {
        "model":       LLM_MODEL_NAME,
        "temperature": temp,
        "api_key":     OPENAI_API_KEY,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)

# ── 벡터 DB 경로 상수 ─────────────────────────────────────────────
PERSIST_TEXT     = str(BASE_DIR / 'vectordb' / 'tricare' / 'chroma_db')
PERSIST_TABLE    = str(BASE_DIR / 'vectordb' / 'tricare' / 'chroma_db2')
COLLECTION_TEXT  = 'tricare_rag'
COLLECTION_TABLE = 'tricare_cost_tables'

LANGUAGE_NAME_MAP = {
    'ko': 'Korean',
    'en': 'English',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'other': "the same language as the user's question",
}

# ── 전역 객체 ─────────────────────────────────────────────────────
embedding_model    = None
vector_store       = None
table_vector_store = None
bm25_retriever     = None
reranker           = None
model              = None
_norm_model        = None
_all_text_chunks   = None


def load_vector_stores(device: str = 'cpu') -> None:
    global embedding_model, vector_store, table_vector_store
    global bm25_retriever, reranker, model, _norm_model, _all_text_chunks

    print('⏳ 모델 및 벡터 DB 로드 중...')

    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': device}
    )
    vector_store = Chroma(
        collection_name=COLLECTION_TEXT,
        embedding_function=embedding_model,
        persist_directory=PERSIST_TEXT
    )
    table_vector_store = Chroma(
        collection_name=COLLECTION_TABLE,
        embedding_function=embedding_model,
        persist_directory=PERSIST_TABLE
    )

    # make_llm()으로 교체 (하드코딩 제거)
    model       = make_llm(temp=0.1)
    _norm_model = make_llm(temp=0)
    reranker    = CrossEncoder('BAAI/bge-reranker-v2-m3')

    raw = vector_store._collection.get(include=['documents', 'metadatas'])
    _all_text_chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw['documents'], raw['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(_all_text_chunks, k=6)

    print(f'✅ 텍스트 벡터 DB: {vector_store._collection.count()}개 벡터')
    print(f'✅ 표 벡터 DB:     {table_vector_store._collection.count()}개 벡터')
    print(f'✅ BM25 인덱스:    {len(_all_text_chunks)}개 청크')
    print('✅ 로드 완료')


# ── 유틸 함수 ─────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    if any('\u4e00' <= c <= '\u9fff' for c in text): return 'zh'
    if any('\u3040' <= c <= '\u30ff' for c in text): return 'ja'
    if any('\uac00' <= c <= '\ud7a3' for c in text): return 'ko'
    return 'en'


def format_docs(docs: list) -> str:
    result = []
    for doc in docs:
        source  = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
        page    = doc.metadata.get('page', '')
        label   = f'[{source}{", p." + str(page) if page else ""}]'
        content = doc.page_content
        if '[search_tags]' in content:
            content = content.split('[search_tags]')[0].strip()
        result.append(f'{label}\n{content}')
    return '\n\n'.join(result)


def normalize_question(question: str) -> dict:
    prompt = (
        "You are a TRICARE insurance query analyzer.\n"
        "Extract:\n"
        "1. intent: one of [coverage, eligibility, cost, pharmacy, dental, overseas, general]\n"
        "2. region: one of [CONUS, OCONUS, korea, unknown]\n"
        "3. english_query: rewrite in clear English (for vector search)\n\n"
        'Respond ONLY in JSON: {{"intent":"...","region":"...","english_query":"..."}}\n\n'
        f"Question: {question}"
    )
    try:
        resp = _norm_model.invoke([HumanMessage(content=prompt)])
        return _json.loads(resp.content)
    except Exception:
        return {'intent': 'general', 'region': 'unknown', 'english_query': question}


# ── Hybrid + Rerank 검색 ──────────────────────────────────────────

def _hybrid_retrieve(question, bm25_ret, vec_ret,
                     bm25_weight=0.4, vec_weight=0.6, k=6):
    bm25_docs = bm25_ret.invoke(question)
    vec_docs  = vec_ret.invoke(question)
    scores, doc_map = {}, {}
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:100]
        scores[key]  = scores.get(key, 0) + bm25_weight * (1 / (rank + 1))
        doc_map[key] = doc
    for rank, doc in enumerate(vec_docs):
        key = doc.page_content[:100]
        scores[key]  = scores.get(key, 0) + vec_weight * (1 / (rank + 1))
        doc_map[key] = doc
    return [doc_map[k_] for k_ in sorted(scores, key=lambda x: scores[x], reverse=True)[:k]]


def hybrid_retrieve_wide(question: str, k: int = 20) -> list:
    bm25_retriever.k = k
    vec_wide = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': k, 'fetch_k': 40}
    )
    result = _hybrid_retrieve(question, bm25_retriever, vec_wide, k=k)
    bm25_retriever.k = 6
    return result


def rerank_docs(question: str, docs: list, top_k: int = 6) -> list:
    if not docs:
        return docs
    pairs = []
    for doc in docs:
        content = doc.page_content
        if '[search_tags]' in content:
            content = content.split('[search_tags]')[0].strip()
        pairs.append((question, content))
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def search(question: str) -> list:
    candidates = hybrid_retrieve_wide(question)
    return rerank_docs(question, candidates, top_k=6)


# ── RAG 체인 (단발성 질문용) ──────────────────────────────────────

def generate_answer(question: str, conversation_context: str = '') -> tuple:
    """
    팀 공통 인터페이스 함수.
    eval_runner.py에서 generate_answer(question) 형태로 호출.
    Returns: (answer: str, docs: list)
    """
    language_code   = detect_language(question)
    answer_language = LANGUAGE_NAME_MAP.get(language_code, 'English')

    docs    = search(question)
    context = format_docs(docs)

    conv_section = (
        f'[이전 대화]\n{conversation_context}\n\n'
        if conversation_context else ''
    )

    prompt_text = (
        'You are a TRICARE health benefits specialist.\n'
        'This system is designed for OCONUS beneficiaries, '
        'primarily Korean residents and USFK (주한미군) personnel.\n\n'
        'IMPORTANT OCONUS RULES:\n'
        '- In South Korea, TRICARE is the PRIMARY payer (not Medicare).\n'
        '- Medicare does NOT cover overseas medical expenses.\n'
        '- Overseas claims require pay-up-front then submit within 3 years.\n'
        '- Medicare Part B must be actively enrolled for overseas residents.\n\n'
        + conv_section +
        'Answer ONLY based on the provided TRICARE documents below.\n'
        'If not in documents, say "The information could not be found in the provided documents."\n'
        'Always mention Group A/B, plan type, beneficiary status when relevant.\n'
        'Do not recommend or suggest enrollment in any specific plan.\n\n'
        'MANDATORY: Every coverage-related answer MUST end with this exact disclaimer:\n'
        '"⚠️ For final plan selection, please consult directly with your personal circumstances and coverage needs."\n\n'
        f'IMPORTANT: Answer in {answer_language}.\n'
        'Term locking: 본인부담금(Copay), 공제액(Deductible), 사전승인(Prior Authorization).\n\n'
        f'[참고 문서]\n{context}\n\n'
        f'[질문]\n{question}\n\n'
        '[답변]\n'
    )

    resp = model.invoke([HumanMessage(content=prompt_text)])
    return resp.content, docs


def make_rag_chain_v3(question: str, conversation_context: str = '') -> tuple:
    """generate_answer의 별칭 — 기존 코드 호환용."""
    return generate_answer(question, conversation_context)
