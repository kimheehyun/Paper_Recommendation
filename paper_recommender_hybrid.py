<<<<<<< HEAD
import streamlit as st
import arxiv
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime
from groq import Groq
import os
import time
from deep_translator import GoogleTranslator

# ========================================
# 번역 함수 (한국어 → 영어)
# ========================================
def to_english(text: str) -> str:
    """자동 감지 후 영어로 번역. 실패 시 원문 그대로 반환"""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ========================================
# Groq API 설정
# ========================================
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=groq_api_key)

# ========================================
# 모델 불러오기 및 페이지 설정
# ========================================
st.set_page_config(page_title="논문 추천 챗봇", layout="wide")
st.title("논문 추천 챗봇")
st.write("arXiv + Semantic Scholar Co-Citation을 활용한 하이브리드 논문 추천 서비스")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ========================================
# 세션 상태 초기화
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "papers_cache" not in st.session_state:
    st.session_state.papers_cache = {}
if "last_papers" not in st.session_state:
    st.session_state.last_papers = None
if "last_scores" not in st.session_state:
    st.session_state.last_scores = None
if "last_semantic_sim" not in st.session_state:
    st.session_state.last_semantic_sim = None
if "last_citations" not in st.session_state:
    st.session_state.last_citations = None
if "last_recency" not in st.session_state:
    st.session_state.last_recency = None
if "last_co_citation" not in st.session_state:
    st.session_state.last_co_citation = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

# ========================================
# arXiv 논문 가져오기
# ========================================
def fetch_arxiv_papers(query, max_results=50):
    """max_results를 50으로 증가 (2단계 필터링을 위해)"""
    try:
        client_arxiv = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        papers = list(client_arxiv.results(search))
        df = pd.DataFrame(
            {
                "title": [p.title for p in papers],
                "abstract": [p.summary if p.summary else "" for p in papers],
                "arxiv_id": [p.entry_id.split("/abs/")[-1] for p in papers],
                "doi": [p.doi if p.doi else "" for p in papers],
                "published": [p.published.strftime("%Y-%m-%d") for p in papers],
                "authors": [
                    ", ".join([author.name for author in p.authors[:3]]) for p in papers
                ],
            }
        )
        return df
    except Exception as e:
        st.error(f"arXiv 검색 오류: {str(e)}")
        return pd.DataFrame()

# ========================================
# Semantic Scholar 정보 가져오기
# ========================================
def fetch_semanticscholar_info(title, arxiv_id):
    cache_key = title + "_" + arxiv_id
    if cache_key in st.session_state.papers_cache:
        return st.session_state.papers_cache[cache_key]

    default_result = {
        "paper_id": "",
        "citation_count": 0,
        "influential_citation_count": 0,
        "publication_date": "",
        "found_by": "not_found"
    }

    def search_ss(query_type, query_value):
        try:
            if query_type == "title":
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": query_value,
                    "limit": 1,
                    "fields": "paperId,citationCount,influentialCitationCount,publicationDate,title"
                }
                res = requests.get(url, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("data") and len(data["data"]) > 0:
                        paper = data["data"][0]
                        return paper, "title"

            elif query_type == "arxiv_id":
                paper_id = f"ARXIV:{query_value}"
                url_id = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                params_id = {
                    "fields": "paperId,citationCount,influentialCitationCount,publicationDate,title"
                }
                res = requests.get(url_id, params=params_id, timeout=5)
                if res.status_code == 200:
                    paper = res.json()
                    if paper.get("paperId"): 
                        return paper, "arxiv_id"
        except Exception:
            return None, None
        return None, None

    paper_info, found_by = search_ss("title", title)
    
    if not paper_info and arxiv_id:
        paper_info, found_by = search_ss("arxiv_id", arxiv_id)

    if paper_info:
        result = {
            "paper_id": paper_info.get("paperId", ""),
            "citation_count": paper_info.get("citationCount", 0),
            "influential_citation_count": paper_info.get("influentialCitationCount", 0),
            "publication_date": paper_info.get("publicationDate", ""),
            "found_by": found_by
        }
        st.session_state.papers_cache[cache_key] = result
        time.sleep(0.15)  # API 제한 완화
        return result
    
    st.session_state.papers_cache[cache_key] = default_result
    return default_result

# ========================================
# 특정 논문의 citations 가져오기 (개선: 에러 핸들링 + 재시도)
# ========================================
def fetch_paper_citations(paper_id, limit=100, max_retries=2):
    """
    특정 논문을 인용한 논문의 ID 리스트 반환 (재시도 로직 추가)
    """
    if not paper_id:
        return []
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            params = {
                "fields": "paperId",
                "limit": limit
            }
            res = requests.get(url, params=params, timeout=8)
            
            if res.status_code == 200:
                data = res.json()
                citing_papers = []
                for item in data.get("data", []):
                    citing_paper = item.get("citingPaper", {})
                    if citing_paper.get("paperId"):
                        citing_papers.append(citing_paper["paperId"])
                
                time.sleep(0.2)  # API 제한 완화
                return citing_papers
                
            elif res.status_code == 429:  # Too Many Requests
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            
    return []

# ========================================
# 공동인용 점수 계산 (개선버전)
# ========================================
def build_co_citation_scores(paper_ids, limit=100, similarity_metric="jaccard", seed_window=5):
    """
    개선된 공동인용 점수 계산
    - API 호출 최소화
    - 에러 핸들링 강화
    - 진행상황 표시
    """
    if not paper_ids:
        return np.zeros(len(paper_ids))

    # 1단계: 시드 논문 선정 (유효한 paper_id가 있는 상위 N개)
    valid_seed_indices = []
    for idx, pid in enumerate(paper_ids[:seed_window]):
        if pid:
            valid_seed_indices.append(idx)
    
    if not valid_seed_indices:
        st.warning("⚠️ 유효한 시드 논문을 찾을 수 없어 공동인용 분석을 건너뜁니다.")
        return np.zeros(len(paper_ids))
    
    # 2단계: 시드 논문들의 인용 정보만 가져오기 (API 호출 최소화)
    st.info(f"🔍 {len(valid_seed_indices)}개 시드 논문의 인용 정보를 수집 중...")
    seed_citation_sets = []
    
    for idx in valid_seed_indices:
        pid = paper_ids[idx]
        citations = fetch_paper_citations(pid, limit=limit)
        
        if citations:
            seed_citation_sets.append(set(citations))
            st.caption(f"   ✓ 시드 논문 {idx+1}: {len(citations)}개 인용 발견")
        else:
            seed_citation_sets.append(set())
            st.caption(f"   ✗ 시드 논문 {idx+1}: 인용 정보 없음")
    
    # 모든 시드의 인용 집합이 비어있으면 종료
    if all(len(s) == 0 for s in seed_citation_sets):
        st.warning("⚠️ 시드 논문들의 인용 정보를 가져올 수 없어 공동인용 분석을 건너뜁니다.")
        return np.zeros(len(paper_ids))
    
    # 3단계: 모든 후보 논문의 인용 정보 수집
    st.info(f"🔍 {len(paper_ids)}개 후보 논문의 인용 정보를 수집 중...")
    all_citation_sets = []
    
    for idx, pid in enumerate(paper_ids):
        if pid:
            # 시드 논문은 이미 수집했으므로 재사용
            if idx in valid_seed_indices:
                seed_idx = valid_seed_indices.index(idx)
                all_citation_sets.append(seed_citation_sets[seed_idx])
            else:
                citations = fetch_paper_citations(pid, limit=limit)
                all_citation_sets.append(set(citations) if citations else set())
        else:
            all_citation_sets.append(set())
        
        # 진행상황 표시 (10개마다)
        if (idx + 1) % 10 == 0:
            st.caption(f"   처리 중: {idx+1}/{len(paper_ids)}")
    
    # 4단계: 각 후보 논문과 시드 논문들 간의 공동인용 유사도 계산
    scores = []
    
    for i in range(len(paper_ids)):
        candidate_set = all_citation_sets[i]
        
        if not candidate_set:
            scores.append(0.0)
            continue
        
        # 시드 논문들과의 평균 유사도 계산
        similarities = []
        for seed_idx in valid_seed_indices:
            if seed_idx == i:  # 자기 자신은 제외
                continue
                
            seed_set = all_citation_sets[seed_idx]
            if not seed_set:
                continue
            
            # Jaccard 유사도 계산
            intersection = len(candidate_set & seed_set)
            union = len(candidate_set | seed_set)
            
            if union > 0:
                if similarity_metric == "cosine":
                    sim = intersection / np.sqrt(len(candidate_set) * len(seed_set))
                else:  # jaccard
                    sim = intersection / union
                similarities.append(sim)
        
        # 평균 유사도
        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        scores.append(avg_sim)
    
    # 5단계: 정규화 (0-1 범위)
    scores = np.array(scores)
    max_score = scores.max()
    
    if max_score > 0:
        scores = scores / max_score
        st.success(f"✓ 공동인용 분석 완료! (최대 유사도: {max_score:.4f})")
    else:
        st.warning("⚠️ 유의미한 공동인용 패턴을 찾지 못했습니다.")
    
    return scores

# ========================================
# 추천 점수 계산 (2단계 필터링 방식)
# ========================================
def calculate_recommendation_score(papers_df, query_embedding, top_n=10, use_two_stage=True):
    """
    use_two_stage=True: 50개 수집 → 인용 기반 필터링 → 15개로 압축 → 정밀 분석
    use_two_stage=False: 기존 방식 (30개 모두 분석)
    """
    papers_df = papers_df.copy()
    papers_df["abstract"] = papers_df["abstract"].fillna("").astype(str)
    
    # 임베딩 계산
    texts = papers_df["title"].astype(str) + ". " + papers_df["abstract"].astype(str)
    embeddings = model.encode(texts.tolist())
    
    # 코사인 유사도
    semantic_scores = cosine_similarity([query_embedding], embeddings)[0]
    
    # ============================================================
    # 1단계: 빠른 필터링 (인용수 + 의미론적 유사도로 후보 압축)
    # ============================================================
    if use_two_stage and len(papers_df) > 15:
        st.info("🔍 1단계: 인용수 기반 사전 필터링 중...")
        
        # Semantic Scholar 정보 가져오기 (빠른 필터링용)
        quick_citation_scores = []
        for idx, row in papers_df.iterrows():
            info = fetch_semanticscholar_info(title=row["title"], arxiv_id=row["arxiv_id"])
            citation_count = info["citation_count"]
            quick_citation_scores.append(citation_count)
        
        # 의미론적 유사도 + 인용수로 1차 점수 계산
        quick_citation_scores = np.array(quick_citation_scores)
        normalized_citations = quick_citation_scores / (quick_citation_scores.max() + 1)
        
        normalized_semantic = semantic_scores / (semantic_scores.max() + 0.001)
        
        # 1차 점수: 의미 + 인용
        quick_scores = 0.8 * normalized_semantic + 0.2 * normalized_citations
        
        # 상위 15개만 선택 (정밀 분석 대상)
        top_15_idx = np.argsort(quick_scores)[::-1][:15]
        papers_df = papers_df.iloc[top_15_idx].reset_index(drop=True)
        semantic_scores = semantic_scores[top_15_idx]
        embeddings = embeddings[top_15_idx]
        
        st.success(f"✓ 상위 15개 후보로 압축 완료 (인용수 범위: {quick_citation_scores[top_15_idx].min():.0f}~{quick_citation_scores[top_15_idx].max():.0f}회)")
    
    # ============================================================
    # 2단계: 정밀 분석 (공동인용 포함)
    # ============================================================
    st.info("🔍 2단계: 정밀 분석 시작...")
    
    # Semantic Scholar 정보 다시 가져오기 (캐시 활용)
    citation_scores = []
    recency_scores = []
    ss_info_list = []
    paper_ids = []
    
    for idx, row in papers_df.iterrows():
        info = fetch_semanticscholar_info(title=row["title"], arxiv_id=row["arxiv_id"]) 
        ss_info_list.append(info)
        paper_ids.append(info["paper_id"])
        
        citation_count = info["citation_count"]
        citation_score = min(citation_count / 100, 1.0) if citation_count > 0 else 0
        citation_scores.append(citation_score)
        
        # 최신성 점수
        pub_date = datetime.strptime(row["published"], "%Y-%m-%d")
        days_old = (datetime.now() - pub_date).days
        recency_score = max(1 - (days_old / 3650), 0)
        recency_scores.append(recency_score)
    
    # 공동 인용 기반 점수 계산
    st.divider()
    co_citation_scores = build_co_citation_scores(
        paper_ids, limit=100, similarity_metric="jaccard", seed_window=5
    )
    
    # 최종 점수 계산
    citation_scores = np.array(citation_scores)
    recency_scores = np.array(recency_scores)
    co_citation_scores = np.array(co_citation_scores)
    
    # semantic_scores 정규화
    if semantic_scores.max() > 0:
        normalized_semantic = semantic_scores / semantic_scores.max()
    else:
        normalized_semantic = semantic_scores
    
    # 최종 가중치: 의미 + 인용 + 최신성 + 공동인용
    final_scores = (
        0.70 * normalized_semantic
        + 0.10 * citation_scores
        + 0.10 * recency_scores
        + 0.10 * co_citation_scores
    )
    
    top_idx = np.argsort(final_scores)[::-1][:top_n]
    result_df = papers_df.iloc[top_idx].reset_index(drop=True)
    result_scores = final_scores[top_idx]
    semantic_sim = semantic_scores[top_idx]
    citations = [citation_scores[i] for i in top_idx]
    recency = [recency_scores[i] for i in top_idx]
    co_citation = [co_citation_scores[i] for i in top_idx]
    
    # 결과에 추가 정보 포함
    result_df["citation_count"] = [ss_info_list[i]["citation_count"] for i in top_idx]
    result_df["found_by"] = [ss_info_list[i]["found_by"] for i in top_idx]
    result_df["co_citation_score"] = co_citation
    
    return result_df, result_scores, semantic_sim, citations, recency, co_citation

# ========================================
# LLM 설명 생성
# ========================================
def generate_recommendation_explanation(user_query, recommended_papers):
    papers_info = ""
    for idx, row in recommended_papers.iterrows():
        co_citation_info = f"\nCo-citation Score: {row.get('co_citation_score', 0):.3f}" if row.get('co_citation_score', 0) > 0 else ""
        
        papers_info += f"\n---\nPaper {idx+1}: {row['title']}\nAuthors: {row['authors']}\nPublished: {row['published']}\nCitations: {row.get('citation_count', 0)}{co_citation_info}\nAbstract: {row['abstract'][:300]}...\n"
    
    prompt = f"""The user is interested in the following field: "{user_query}"

Analyze the abstracts of the recommended papers below. The 'Co-citation Score' reflects how frequently the candidate is cited together with the seed papers across the literature. Provide a concise abstract summary and a professional explanation of why the paper was recommended.

The output MUST be in Korean and strictly follow the format below. Separate the analysis of each paper using the exact phrase: ###END_OF_PAPER_ANALYSIS###

{papers_info}

Format:
- 초록 요약 [N]: [간단한 설명] \n
- 논문 추천 근거 [N]: [간단한 설명]
###END_OF_PAPER_ANALYSIS###
"""
    
    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return message.choices[0].message.content
    except Exception as e:
        return f"LLM 설명 생성 오류: {str(e)}"

# ========================================
# 챗봇 입력 처리
# ========================================
def chat_with_user(user_input):
    # 🟢 Step 1. 입력을 영어로 번역
    user_input_en = to_english(user_input)

    with st.spinner("지금 arXiv에서 관련 논문을 검색하고 있습니다..."):
        # 🟢 Step 2. 영어 번역된 쿼리로 검색
        papers_df = fetch_arxiv_papers(user_input_en, max_results=50)

    if papers_df.empty:
        response = "죄송합니다. 해당 주제의 논문을 찾을 수 없습니다. 다른 키워드로 시도해 주세요."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        return

    # 🟢 Step 3. 영어 쿼리로 임베딩 계산
    query_embedding = model.encode(user_input_en)

    with st.spinner("지금 Semantic Scholar에서 인용 정보 및 공동인용 분석 중..."):
        rec_papers, scores, semantic_sim, citations, recency, co_citation = (
            calculate_recommendation_score(papers_df, query_embedding, top_n=5)
        )

    with st.spinner("지금 LLM이 추천 이유를 분석하고 있습니다..."):
        explanation = generate_recommendation_explanation(user_input, rec_papers)

    # 🟢 Step 4. 안내 문구에 번역 내용 표시
    response = (
        f"**'{user_input}'** → **'{user_input_en}'**(영어 번역)으로 검색했습니다.\n"
        f"관련 추천 논문 {len(rec_papers)}개를 찾았습니다. 아래에서 상세 정보를 확인해 주세요."
    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 결과 저장 및 화면 갱신은 동일
    st.session_state.last_papers = rec_papers
    st.session_state.last_scores = scores
    st.session_state.last_semantic_sim = semantic_sim
    st.session_state.last_citations = citations
    st.session_state.last_recency = recency
    st.session_state.last_co_citation = co_citation
    st.session_state.last_explanation = explanation

    st.rerun()

# ========================================
# UI 레이아웃
# ========================================

message_count = len(st.session_state.messages)
if message_count > 0:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_query = st.chat_input("관심 있는 분야나 논문 주제를 입력하세요.")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    chat_with_user(user_query)

# 최신 추천 논문 상세 정보
if st.session_state.last_papers is not None and not st.session_state.last_papers.empty:
    rec_papers = st.session_state.last_papers
    scores = st.session_state.last_scores
    semantic_sim = st.session_state.last_semantic_sim
    citations = st.session_state.last_citations
    recency = st.session_state.last_recency
    co_citation = st.session_state.last_co_citation
    
    st.divider()
    st.subheader("최신 추천 논문 상세 정보")
    
    for idx, row in rec_papers.iterrows():
        with st.expander(f"**{idx+1}. {row['title']}**"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**저자:** {row['authors']}")
                st.write(f"**발표일:** {row['published']}")
                st.write(f"**arXiv ID:** {row['arxiv_id']}")
                if row.get('doi'):
                    st.write(f"**DOI:** {row['doi']}")
                st.write(f"**인용수:** {row.get('citation_count', 0)}회")
                st.write(f"**검색방법:** {row.get('found_by', 'N/A')}")
                st.write(f"\n**초록:**\n{row['abstract']}")
            with col2:
                st.metric("추천 점수", f"{scores[idx]:.3f}")
                st.metric("의미론적 유사도", f"{semantic_sim[idx]:.3f}")
                st.metric("인용 기반 점수", f"{citations[idx]:.3f}")
                st.metric("최신성 점수", f"{recency[idx]:.3f}")
                st.metric("공동 인용 점수", f"{co_citation[idx]:.3f}",
                        help="시드 논문들과 함께 인용되는 빈도를 기반으로 계산한 유사도입니다.")
            
            paper_url = f"https://arxiv.org/abs/{row['arxiv_id']}"
            st.markdown(f"[arXiv에서 보기]({paper_url})")

# LLM 분석 결과
if st.session_state.last_explanation:
    st.divider()
    st.subheader("최신 LLM 논문 초록 요약 및 추천 분석")
    
    analysis_parts = st.session_state.last_explanation.split("###END_OF_PAPER_ANALYSIS###")
    
    for i, part in enumerate(analysis_parts):
        cleaned_part = part.strip()
        if cleaned_part:
            if i > 0:
                st.divider()
=======
import streamlit as st
import arxiv
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime
from groq import Groq
import os
import time
from deep_translator import GoogleTranslator

# ========================================
# 번역 함수 (한국어 → 영어)
# ========================================
def to_english(text: str) -> str:
    """자동 감지 후 영어로 번역. 실패 시 원문 그대로 반환"""
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ========================================
# Groq API 설정
# ========================================
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=groq_api_key)

# ========================================
# 모델 불러오기 및 페이지 설정
# ========================================
st.set_page_config(page_title="논문 추천 챗봇", layout="wide")
st.title("논문 추천 챗봇")
st.write("arXiv + Semantic Scholar Co-Citation을 활용한 하이브리드 논문 추천 서비스")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = load_model()

# ========================================
# 세션 상태 초기화
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "papers_cache" not in st.session_state:
    st.session_state.papers_cache = {}
if "last_papers" not in st.session_state:
    st.session_state.last_papers = None
if "last_scores" not in st.session_state:
    st.session_state.last_scores = None
if "last_semantic_sim" not in st.session_state:
    st.session_state.last_semantic_sim = None
if "last_citations" not in st.session_state:
    st.session_state.last_citations = None
if "last_recency" not in st.session_state:
    st.session_state.last_recency = None
if "last_co_citation" not in st.session_state:
    st.session_state.last_co_citation = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

# ========================================
# arXiv 논문 가져오기
# ========================================
def fetch_arxiv_papers(query, max_results=50):
    """max_results를 50으로 증가 (2단계 필터링을 위해)"""
    try:
        client_arxiv = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        papers = list(client_arxiv.results(search))
        df = pd.DataFrame(
            {
                "title": [p.title for p in papers],
                "abstract": [p.summary if p.summary else "" for p in papers],
                "arxiv_id": [p.entry_id.split("/abs/")[-1] for p in papers],
                "doi": [p.doi if p.doi else "" for p in papers],
                "published": [p.published.strftime("%Y-%m-%d") for p in papers],
                "authors": [
                    ", ".join([author.name for author in p.authors[:3]]) for p in papers
                ],
            }
        )
        return df
    except Exception as e:
        st.error(f"arXiv 검색 오류: {str(e)}")
        return pd.DataFrame()

# ========================================
# Semantic Scholar 정보 가져오기
# ========================================
def fetch_semanticscholar_info(title, arxiv_id):
    cache_key = title + "_" + arxiv_id
    if cache_key in st.session_state.papers_cache:
        return st.session_state.papers_cache[cache_key]

    default_result = {
        "paper_id": "",
        "citation_count": 0,
        "influential_citation_count": 0,
        "publication_date": "",
        "found_by": "not_found"
    }

    def search_ss(query_type, query_value):
        try:
            if query_type == "title":
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": query_value,
                    "limit": 1,
                    "fields": "paperId,citationCount,influentialCitationCount,publicationDate,title"
                }
                res = requests.get(url, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("data") and len(data["data"]) > 0:
                        paper = data["data"][0]
                        return paper, "title"

            elif query_type == "arxiv_id":
                paper_id = f"ARXIV:{query_value}"
                url_id = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                params_id = {
                    "fields": "paperId,citationCount,influentialCitationCount,publicationDate,title"
                }
                res = requests.get(url_id, params=params_id, timeout=5)
                if res.status_code == 200:
                    paper = res.json()
                    if paper.get("paperId"): 
                        return paper, "arxiv_id"
        except Exception:
            return None, None
        return None, None

    paper_info, found_by = search_ss("title", title)
    
    if not paper_info and arxiv_id:
        paper_info, found_by = search_ss("arxiv_id", arxiv_id)

    if paper_info:
        result = {
            "paper_id": paper_info.get("paperId", ""),
            "citation_count": paper_info.get("citationCount", 0),
            "influential_citation_count": paper_info.get("influentialCitationCount", 0),
            "publication_date": paper_info.get("publicationDate", ""),
            "found_by": found_by
        }
        st.session_state.papers_cache[cache_key] = result
        time.sleep(0.15)  # API 제한 완화
        return result
    
    st.session_state.papers_cache[cache_key] = default_result
    return default_result

# ========================================
# 특정 논문의 citations 가져오기 (개선: 에러 핸들링 + 재시도)
# ========================================
def fetch_paper_citations(paper_id, limit=100, max_retries=2):
    """
    특정 논문을 인용한 논문의 ID 리스트 반환 (재시도 로직 추가)
    """
    if not paper_id:
        return []
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            params = {
                "fields": "paperId",
                "limit": limit
            }
            res = requests.get(url, params=params, timeout=8)
            
            if res.status_code == 200:
                data = res.json()
                citing_papers = []
                for item in data.get("data", []):
                    citing_paper = item.get("citingPaper", {})
                    if citing_paper.get("paperId"):
                        citing_papers.append(citing_paper["paperId"])
                
                time.sleep(0.2)  # API 제한 완화
                return citing_papers
                
            elif res.status_code == 429:  # Too Many Requests
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            
    return []

# ========================================
# 공동인용 점수 계산 (개선버전)
# ========================================
def build_co_citation_scores(paper_ids, limit=100, similarity_metric="jaccard", seed_window=5):
    """
    개선된 공동인용 점수 계산
    - API 호출 최소화
    - 에러 핸들링 강화
    - 진행상황 표시
    """
    if not paper_ids:
        return np.zeros(len(paper_ids))

    # 1단계: 시드 논문 선정 (유효한 paper_id가 있는 상위 N개)
    valid_seed_indices = []
    for idx, pid in enumerate(paper_ids[:seed_window]):
        if pid:
            valid_seed_indices.append(idx)
    
    if not valid_seed_indices:
        st.warning("⚠️ 유효한 시드 논문을 찾을 수 없어 공동인용 분석을 건너뜁니다.")
        return np.zeros(len(paper_ids))
    
    # 2단계: 시드 논문들의 인용 정보만 가져오기 (API 호출 최소화)
    st.info(f"🔍 {len(valid_seed_indices)}개 시드 논문의 인용 정보를 수집 중...")
    seed_citation_sets = []
    
    for idx in valid_seed_indices:
        pid = paper_ids[idx]
        citations = fetch_paper_citations(pid, limit=limit)
        
        if citations:
            seed_citation_sets.append(set(citations))
            st.caption(f"   ✓ 시드 논문 {idx+1}: {len(citations)}개 인용 발견")
        else:
            seed_citation_sets.append(set())
            st.caption(f"   ✗ 시드 논문 {idx+1}: 인용 정보 없음")
    
    # 모든 시드의 인용 집합이 비어있으면 종료
    if all(len(s) == 0 for s in seed_citation_sets):
        st.warning("⚠️ 시드 논문들의 인용 정보를 가져올 수 없어 공동인용 분석을 건너뜁니다.")
        return np.zeros(len(paper_ids))
    
    # 3단계: 모든 후보 논문의 인용 정보 수집
    st.info(f"🔍 {len(paper_ids)}개 후보 논문의 인용 정보를 수집 중...")
    all_citation_sets = []
    
    for idx, pid in enumerate(paper_ids):
        if pid:
            # 시드 논문은 이미 수집했으므로 재사용
            if idx in valid_seed_indices:
                seed_idx = valid_seed_indices.index(idx)
                all_citation_sets.append(seed_citation_sets[seed_idx])
            else:
                citations = fetch_paper_citations(pid, limit=limit)
                all_citation_sets.append(set(citations) if citations else set())
        else:
            all_citation_sets.append(set())
        
        # 진행상황 표시 (10개마다)
        if (idx + 1) % 10 == 0:
            st.caption(f"   처리 중: {idx+1}/{len(paper_ids)}")
    
    # 4단계: 각 후보 논문과 시드 논문들 간의 공동인용 유사도 계산
    scores = []
    
    for i in range(len(paper_ids)):
        candidate_set = all_citation_sets[i]
        
        if not candidate_set:
            scores.append(0.0)
            continue
        
        # 시드 논문들과의 평균 유사도 계산
        similarities = []
        for seed_idx in valid_seed_indices:
            if seed_idx == i:  # 자기 자신은 제외
                continue
                
            seed_set = all_citation_sets[seed_idx]
            if not seed_set:
                continue
            
            # Jaccard 유사도 계산
            intersection = len(candidate_set & seed_set)
            union = len(candidate_set | seed_set)
            
            if union > 0:
                if similarity_metric == "cosine":
                    sim = intersection / np.sqrt(len(candidate_set) * len(seed_set))
                else:  # jaccard
                    sim = intersection / union
                similarities.append(sim)
        
        # 평균 유사도
        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        scores.append(avg_sim)
    
    # 5단계: 정규화 (0-1 범위)
    scores = np.array(scores)
    max_score = scores.max()
    
    if max_score > 0:
        scores = scores / max_score
        st.success(f"✓ 공동인용 분석 완료! (최대 유사도: {max_score:.4f})")
    else:
        st.warning("⚠️ 유의미한 공동인용 패턴을 찾지 못했습니다.")
    
    return scores

# ========================================
# 추천 점수 계산 (2단계 필터링 방식)
# ========================================
def calculate_recommendation_score(papers_df, query_embedding, top_n=10, use_two_stage=True):
    """
    use_two_stage=True: 50개 수집 → 인용 기반 필터링 → 15개로 압축 → 정밀 분석
    use_two_stage=False: 기존 방식 (30개 모두 분석)
    """
    papers_df = papers_df.copy()
    papers_df["abstract"] = papers_df["abstract"].fillna("").astype(str)
    
    # 임베딩 계산
    texts = papers_df["title"].astype(str) + ". " + papers_df["abstract"].astype(str)
    embeddings = model.encode(texts.tolist())
    
    # 코사인 유사도
    semantic_scores = cosine_similarity([query_embedding], embeddings)[0]
    
    # ============================================================
    # 1단계: 빠른 필터링 (인용수 + 의미론적 유사도로 후보 압축)
    # ============================================================
    if use_two_stage and len(papers_df) > 15:
        st.info("🔍 1단계: 인용수 기반 사전 필터링 중...")
        
        # Semantic Scholar 정보 가져오기 (빠른 필터링용)
        quick_citation_scores = []
        for idx, row in papers_df.iterrows():
            info = fetch_semanticscholar_info(title=row["title"], arxiv_id=row["arxiv_id"])
            citation_count = info["citation_count"]
            quick_citation_scores.append(citation_count)
        
        # 의미론적 유사도 + 인용수로 1차 점수 계산
        quick_citation_scores = np.array(quick_citation_scores)
        normalized_citations = quick_citation_scores / (quick_citation_scores.max() + 1)
        
        normalized_semantic = semantic_scores / (semantic_scores.max() + 0.001)
        
        # 1차 점수: 의미 + 인용
        quick_scores = 0.8 * normalized_semantic + 0.2 * normalized_citations
        
        # 상위 15개만 선택 (정밀 분석 대상)
        top_15_idx = np.argsort(quick_scores)[::-1][:15]
        papers_df = papers_df.iloc[top_15_idx].reset_index(drop=True)
        semantic_scores = semantic_scores[top_15_idx]
        embeddings = embeddings[top_15_idx]
        
        st.success(f"✓ 상위 15개 후보로 압축 완료 (인용수 범위: {quick_citation_scores[top_15_idx].min():.0f}~{quick_citation_scores[top_15_idx].max():.0f}회)")
    
    # ============================================================
    # 2단계: 정밀 분석 (공동인용 포함)
    # ============================================================
    st.info("🔍 2단계: 정밀 분석 시작...")
    
    # Semantic Scholar 정보 다시 가져오기 (캐시 활용)
    citation_scores = []
    recency_scores = []
    ss_info_list = []
    paper_ids = []
    
    for idx, row in papers_df.iterrows():
        info = fetch_semanticscholar_info(title=row["title"], arxiv_id=row["arxiv_id"]) 
        ss_info_list.append(info)
        paper_ids.append(info["paper_id"])
        
        citation_count = info["citation_count"]
        citation_score = min(citation_count / 100, 1.0) if citation_count > 0 else 0
        citation_scores.append(citation_score)
        
        # 최신성 점수
        pub_date = datetime.strptime(row["published"], "%Y-%m-%d")
        days_old = (datetime.now() - pub_date).days
        recency_score = max(1 - (days_old / 3650), 0)
        recency_scores.append(recency_score)
    
    # 공동 인용 기반 점수 계산
    st.divider()
    co_citation_scores = build_co_citation_scores(
        paper_ids, limit=100, similarity_metric="jaccard", seed_window=5
    )
    
    # 최종 점수 계산
    citation_scores = np.array(citation_scores)
    recency_scores = np.array(recency_scores)
    co_citation_scores = np.array(co_citation_scores)
    
    # semantic_scores 정규화
    if semantic_scores.max() > 0:
        normalized_semantic = semantic_scores / semantic_scores.max()
    else:
        normalized_semantic = semantic_scores
    
    # 최종 가중치: 의미 + 인용 + 최신성 + 공동인용
    final_scores = (
        0.70 * normalized_semantic
        + 0.10 * citation_scores
        + 0.10 * recency_scores
        + 0.10 * co_citation_scores
    )
    
    top_idx = np.argsort(final_scores)[::-1][:top_n]
    result_df = papers_df.iloc[top_idx].reset_index(drop=True)
    result_scores = final_scores[top_idx]
    semantic_sim = semantic_scores[top_idx]
    citations = [citation_scores[i] for i in top_idx]
    recency = [recency_scores[i] for i in top_idx]
    co_citation = [co_citation_scores[i] for i in top_idx]
    
    # 결과에 추가 정보 포함
    result_df["citation_count"] = [ss_info_list[i]["citation_count"] for i in top_idx]
    result_df["found_by"] = [ss_info_list[i]["found_by"] for i in top_idx]
    result_df["co_citation_score"] = co_citation
    
    return result_df, result_scores, semantic_sim, citations, recency, co_citation

# ========================================
# LLM 설명 생성
# ========================================
def generate_recommendation_explanation(user_query, recommended_papers):
    papers_info = ""
    for idx, row in recommended_papers.iterrows():
        co_citation_info = f"\nCo-citation Score: {row.get('co_citation_score', 0):.3f}" if row.get('co_citation_score', 0) > 0 else ""
        
        papers_info += f"\n---\nPaper {idx+1}: {row['title']}\nAuthors: {row['authors']}\nPublished: {row['published']}\nCitations: {row.get('citation_count', 0)}{co_citation_info}\nAbstract: {row['abstract'][:300]}...\n"
    
    prompt = f"""The user is interested in the following field: "{user_query}"

Analyze the abstracts of the recommended papers below. The 'Co-citation Score' reflects how frequently the candidate is cited together with the seed papers across the literature. Provide a concise abstract summary and a professional explanation of why the paper was recommended.

The output MUST be in Korean and strictly follow the format below. Separate the analysis of each paper using the exact phrase: ###END_OF_PAPER_ANALYSIS###

{papers_info}

Format:
- 초록 요약 [N]: [간단한 설명] \n
- 논문 추천 근거 [N]: [간단한 설명]
###END_OF_PAPER_ANALYSIS###
"""
    
    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return message.choices[0].message.content
    except Exception as e:
        return f"LLM 설명 생성 오류: {str(e)}"

# ========================================
# 챗봇 입력 처리
# ========================================
def chat_with_user(user_input):
    # 🟢 Step 1. 입력을 영어로 번역
    user_input_en = to_english(user_input)

    with st.spinner("지금 arXiv에서 관련 논문을 검색하고 있습니다..."):
        # 🟢 Step 2. 영어 번역된 쿼리로 검색
        papers_df = fetch_arxiv_papers(user_input_en, max_results=50)

    if papers_df.empty:
        response = "죄송합니다. 해당 주제의 논문을 찾을 수 없습니다. 다른 키워드로 시도해 주세요."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
        return

    # 🟢 Step 3. 영어 쿼리로 임베딩 계산
    query_embedding = model.encode(user_input_en)

    with st.spinner("지금 Semantic Scholar에서 인용 정보 및 공동인용 분석 중..."):
        rec_papers, scores, semantic_sim, citations, recency, co_citation = (
            calculate_recommendation_score(papers_df, query_embedding, top_n=5)
        )

    with st.spinner("지금 LLM이 추천 이유를 분석하고 있습니다..."):
        explanation = generate_recommendation_explanation(user_input, rec_papers)

    # 🟢 Step 4. 안내 문구에 번역 내용 표시
    response = (
        f"**'{user_input}'** → **'{user_input_en}'**(영어 번역)으로 검색했습니다.\n"
        f"관련 추천 논문 {len(rec_papers)}개를 찾았습니다. 아래에서 상세 정보를 확인해 주세요."
    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 결과 저장 및 화면 갱신은 동일
    st.session_state.last_papers = rec_papers
    st.session_state.last_scores = scores
    st.session_state.last_semantic_sim = semantic_sim
    st.session_state.last_citations = citations
    st.session_state.last_recency = recency
    st.session_state.last_co_citation = co_citation
    st.session_state.last_explanation = explanation

    st.rerun()

# ========================================
# UI 레이아웃
# ========================================

message_count = len(st.session_state.messages)
if message_count > 0:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_query = st.chat_input("관심 있는 분야나 논문 주제를 입력하세요.")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    chat_with_user(user_query)

# 최신 추천 논문 상세 정보
if st.session_state.last_papers is not None and not st.session_state.last_papers.empty:
    rec_papers = st.session_state.last_papers
    scores = st.session_state.last_scores
    semantic_sim = st.session_state.last_semantic_sim
    citations = st.session_state.last_citations
    recency = st.session_state.last_recency
    co_citation = st.session_state.last_co_citation
    
    st.divider()
    st.subheader("최신 추천 논문 상세 정보")
    
    for idx, row in rec_papers.iterrows():
        with st.expander(f"**{idx+1}. {row['title']}**"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**저자:** {row['authors']}")
                st.write(f"**발표일:** {row['published']}")
                st.write(f"**arXiv ID:** {row['arxiv_id']}")
                if row.get('doi'):
                    st.write(f"**DOI:** {row['doi']}")
                st.write(f"**인용수:** {row.get('citation_count', 0)}회")
                st.write(f"**검색방법:** {row.get('found_by', 'N/A')}")
                st.write(f"\n**초록:**\n{row['abstract']}")
            with col2:
                st.metric("추천 점수", f"{scores[idx]:.3f}")
                st.metric("의미론적 유사도", f"{semantic_sim[idx]:.3f}")
                st.metric("인용 기반 점수", f"{citations[idx]:.3f}")
                st.metric("최신성 점수", f"{recency[idx]:.3f}")
                st.metric("공동 인용 점수", f"{co_citation[idx]:.3f}",
                        help="시드 논문들과 함께 인용되는 빈도를 기반으로 계산한 유사도입니다.")
            
            paper_url = f"https://arxiv.org/abs/{row['arxiv_id']}"
            st.markdown(f"[arXiv에서 보기]({paper_url})")

# LLM 분석 결과
if st.session_state.last_explanation:
    st.divider()
    st.subheader("최신 LLM 논문 초록 요약 및 추천 분석")
    
    analysis_parts = st.session_state.last_explanation.split("###END_OF_PAPER_ANALYSIS###")
    
    for i, part in enumerate(analysis_parts):
        cleaned_part = part.strip()
        if cleaned_part:
            if i > 0:
                st.divider()
>>>>>>> ddc311f (first commit)
            st.markdown(cleaned_part)