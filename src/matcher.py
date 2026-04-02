"""
RAG Job Matcher - Uses Sentence-BERT embeddings + ChromaDB for semantic
retrieval, then Claude for detailed match scoring and analysis.

This is the core RAG pipeline:
1. Embed resume and job descriptions using Sentence-BERT
2. Store job embeddings in ChromaDB for fast similarity search
3. Retrieve top-k semantically similar jobs
4. Use Claude to do detailed match scoring with skill gap analysis
"""

import os
import json
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic


# Load embedding model (cached after first load)
_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_chroma_collection(collection_name: str = "jobs"):
    """Get or create a ChromaDB collection."""
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Delete existing collection to avoid stale data
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _build_job_text(job: dict) -> str:
    """Build a single text representation of a job for embedding."""
    parts = [
        f"Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
    ]
    if job.get("salary"):
        parts.append(f"Salary: {job['salary']}")
    parts.append(f"Description: {job.get('description', '')}")
    return "\n".join(parts)


def _job_id(job: dict, index: int) -> str:
    """Generate a stable ID for a job."""
    raw = f"{job.get('title', '')}-{job.get('company', '')}-{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


MATCH_PROMPT = """You are an expert job matching analyst. Analyze how well this candidate's resume matches each job posting.

CANDIDATE RESUME:
{resume_text}

CANDIDATE PROFILE (extracted):
{profile_json}

JOB POSTINGS TO ANALYZE:
{jobs_text}

For EACH job, provide a detailed analysis. Return ONLY valid JSON (no markdown, no backticks) as a list:

[
    {{
        "job_index": 0,
        "score": 78,
        "matching_skills": ["Python", "SQL", "Power BI"],
        "skill_gaps": ["Tableau", "Spark"],
        "reasoning": "Strong match on core data analyst skills. Candidate's experience with 2.1M record datasets and Power BI dashboards directly aligns. Gap in Spark which is listed as preferred but not required."
    }},
    ...
]

SCORING GUIDELINES:
- 85-100: Almost perfect match. Candidate meets all required skills and most preferred ones.
- 70-84: Strong match. Meets most requirements, minor gaps in preferred skills.
- 50-69: Moderate match. Has core skills but missing some requirements.
- 30-49: Weak match. Significant skill gaps but transferable skills exist.
- 0-29: Poor match. Most requirements not met.

Be realistic and specific. Don't inflate scores. Reference specific skills and experiences from the resume.
"""


def match_jobs(
    resume_text: str,
    resume_profile: Optional[dict],
    jobs: list[dict],
    top_k: int = 0,
) -> list[dict]:
    """
    Match resume against job postings using RAG pipeline.

    1. Embed all jobs and resume into vector space
    2. Use ChromaDB to find semantically similar jobs
    3. Use Claude for detailed match analysis

    Args:
        resume_text: Raw resume text
        resume_profile: Extracted profile dict (optional)
        jobs: List of job posting dicts
        top_k: Number of top matches to analyze in detail (0 = all)

    Returns:
        List of match result dicts sorted by score
    """
    if not jobs:
        return []

    model = _get_embed_model()

    # --- Step 1: Embed resume ---
    resume_embedding = model.encode(resume_text[:8000], show_progress_bar=False).tolist()

    # --- Step 2: Embed and store jobs in ChromaDB ---
    collection = _get_chroma_collection()

    job_texts = [_build_job_text(j) for j in jobs]
    job_ids = [_job_id(j, i) for i, j in enumerate(jobs)]
    job_embeddings = model.encode(job_texts, show_progress_bar=False).tolist()

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(jobs), batch_size):
        end = min(i + batch_size, len(jobs))
        collection.add(
            ids=job_ids[i:end],
            embeddings=job_embeddings[i:end],
            documents=job_texts[i:end],
            metadatas=[{"index": idx} for idx in range(i, end)],
        )

    # --- Step 3: Query ChromaDB for most similar jobs ---
    n_results = min(top_k if top_k > 0 else len(jobs), len(jobs))
    query_results = collection.query(
        query_embeddings=[resume_embedding],
        n_results=n_results,
    )

    # Get the indices of matched jobs (ordered by similarity)
    matched_indices = [m["index"] for m in query_results["metadatas"][0]]
    matched_jobs = [jobs[i] for i in matched_indices]
    similarity_scores = query_results["distances"][0] if query_results.get("distances") else []

    # --- Step 4: Use Claude for detailed analysis ---
    # Process in batches of 5 to stay within token limits
    all_results = []
    batch_size = 5

    for batch_start in range(0, len(matched_jobs), batch_size):
        batch_end = min(batch_start + batch_size, len(matched_jobs))
        batch_jobs = matched_jobs[batch_start:batch_end]
        batch_indices = matched_indices[batch_start:batch_end]

        # Format jobs for the prompt
        jobs_text = ""
        for i, job in enumerate(batch_jobs):
            desc = job.get("description", "")[:2000]
            jobs_text += (
                f"\n--- JOB {i} ---\n"
                f"Title: {job.get('title', 'N/A')}\n"
                f"Company: {job.get('company', 'N/A')}\n"
                f"Location: {job.get('location', '')}\n"
                f"Description: {desc}\n"
            )

        profile_json = json.dumps(resume_profile, indent=2) if resume_profile else "{}"

        prompt = MATCH_PROMPT.format(
            resume_text=resume_text[:4000],
            profile_json=profile_json,
            jobs_text=jobs_text,
        )

        try:
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text.strip()

            # Parse JSON response
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[-1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            response_text = response_text.strip()

            # Find JSON array
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end > start:
                batch_results = json.loads(response_text[start:end])
            else:
                batch_results = json.loads(response_text)

            # Merge Claude's analysis with job data
            for analysis in batch_results:
                job_idx = analysis.get("job_index", 0)
                if job_idx < len(batch_jobs):
                    job = batch_jobs[job_idx]
                    result = {
                        **job,
                        "score": analysis.get("score", 50),
                        "matching_skills": analysis.get("matching_skills", []),
                        "skill_gaps": analysis.get("skill_gaps", []),
                        "reasoning": analysis.get("reasoning", ""),
                    }
                    all_results.append(result)

        except Exception as e:
            # Fallback: use embedding similarity as score
            for i, job in enumerate(batch_jobs):
                sim = 1 - similarity_scores[batch_start + i] if similarity_scores else 0.5
                result = {
                    **job,
                    "score": round(sim * 100),
                    "matching_skills": [],
                    "skill_gaps": [],
                    "reasoning": f"Score based on embedding similarity only. LLM analysis failed: {e}",
                }
                all_results.append(result)

    # Sort by score descending
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results
