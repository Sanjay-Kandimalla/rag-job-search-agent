"""
RAG Job Search Agent - AI-Powered Job Matching & Cover Letter Generator

Upload your resume, search for jobs, and get AI-powered match scores,
skill gap analysis, and tailored cover letter drafts.

Built with: Streamlit, LangChain, Anthropic Claude, ChromaDB, Sentence-BERT
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="RAG Job Search Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .match-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .match-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .score-high { background: #dcfce7; color: #166534; }
    .score-mid { background: #fef9c3; color: #854d0e; }
    .score-low { background: #fee2e2; color: #991b1b; }

    .skill-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    .skill-match { background: #dbeafe; color: #1e40af; }
    .skill-gap { background: #fce7f3; color: #9d174d; }

    .metric-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e8e0f0;
    }

    .metric-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] label {
        color: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "resume_text": None,
        "resume_profile": None,
        "jobs_data": [],
        "match_results": [],
        "cover_letters": {},
        "api_key_set": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # API Key
        anthropic_key = st.text_input(
            "🔑 Anthropic API Key",
            type="password",
            help="Get your key at console.anthropic.com",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            st.session_state.api_key_set = True

        jsearch_key = st.text_input(
            "🔍 JSearch API Key (optional)",
            type="password",
            help="Get free key at rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch",
            value=os.environ.get("JSEARCH_API_KEY", ""),
        )
        if jsearch_key:
            os.environ["JSEARCH_API_KEY"] = jsearch_key

        st.markdown("---")
        st.markdown("### 📊 Session Stats")

        col1, col2 = st.columns(2)
        with col1:
            resume_status = "✅" if st.session_state.resume_text else "❌"
            st.markdown(f"Resume: {resume_status}")
        with col2:
            job_count = len(st.session_state.jobs_data)
            st.markdown(f"Jobs: **{job_count}**")

        matches = len(st.session_state.match_results)
        st.markdown(f"Matches: **{matches}**")

        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.75rem; color:#94a3b8;'>"
            "Built by Sanjay Kandimalla<br>"
            "Python · LangChain · Claude · ChromaDB</p>",
            unsafe_allow_html=True,
        )


def render_resume_upload():
    """Render the resume upload section."""
    st.markdown("### 📄 Step 1: Upload Your Resume")

    upload_col, preview_col = st.columns([1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT resume",
            type=["pdf", "txt"],
            help="We extract text, skills, and experience from your resume",
        )

        if uploaded_file and not st.session_state.resume_text:
            with st.spinner("🔍 Parsing your resume..."):
                try:
                    from src.resume_parser import parse_resume

                    text = parse_resume(uploaded_file)
                    st.session_state.resume_text = text

                    if st.session_state.api_key_set:
                        from src.profile_extractor import extract_profile

                        profile = extract_profile(text)
                        st.session_state.resume_profile = profile
                        st.success("Resume parsed & profile extracted!")
                    else:
                        st.warning("Add your Anthropic API key to extract skills profile.")
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")

    with preview_col:
        if st.session_state.resume_profile:
            profile = st.session_state.resume_profile
            st.markdown("**Extracted Profile:**")

            if profile.get("skills"):
                skills_html = " ".join(
                    f'<span class="skill-tag skill-match">{s}</span>'
                    for s in profile["skills"][:20]
                )
                st.markdown(skills_html, unsafe_allow_html=True)

            if profile.get("experience_years"):
                st.markdown(f"**Experience:** {profile['experience_years']}")
            if profile.get("education"):
                st.markdown(f"**Education:** {profile['education']}")
            if profile.get("job_titles"):
                st.markdown(f"**Target Roles:** {', '.join(profile['job_titles'][:3])}")

        elif st.session_state.resume_text:
            st.markdown("**Raw Text Preview:**")
            st.text_area(
                "resume_preview",
                st.session_state.resume_text[:500] + "...",
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )


def render_job_search():
    """Render the job search section."""
    st.markdown("### 🔎 Step 2: Find Jobs")

    tab_api, tab_csv = st.tabs(["🌐 Search Live Jobs", "📁 Upload Job CSV"])

    with tab_api:
        search_col, filter_col = st.columns([2, 1])
        with search_col:
            query = st.text_input(
                "Job search query",
                placeholder="e.g., Data Analyst, Data Scientist, ML Engineer",
                help="Enter job titles, keywords, or skills",
            )
        with filter_col:
            location = st.text_input(
                "Location (optional)",
                placeholder="e.g., Texas, Remote",
            )

        num_results = st.slider("Number of results", 5, 50, 20)

        if st.button("🔍 Search Jobs", type="primary", use_container_width=True):
            if not os.environ.get("JSEARCH_API_KEY"):
                st.error("Please add your JSearch API key in the sidebar.")
            elif not query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("🌐 Searching for jobs..."):
                    try:
                        from src.job_fetcher import search_jobs_api

                        jobs = search_jobs_api(query, location, num_results)
                        st.session_state.jobs_data = jobs
                        st.success(f"Found {len(jobs)} jobs!")
                    except Exception as e:
                        st.error(f"Search error: {e}")

    with tab_csv:
        st.markdown(
            "Upload a CSV with columns: `title`, `company`, `description`, "
            "`location` (optional: `url`, `salary`)"
        )
        csv_file = st.file_uploader("Upload job postings CSV", type=["csv"])
        if csv_file:
            try:
                from src.job_fetcher import parse_jobs_csv

                jobs = parse_jobs_csv(csv_file)
                st.session_state.jobs_data = jobs
                st.success(f"Loaded {len(jobs)} jobs from CSV!")
            except Exception as e:
                st.error(f"CSV error: {e}")

    # Show loaded jobs summary
    if st.session_state.jobs_data:
        st.markdown(f"**{len(st.session_state.jobs_data)} jobs loaded**")
        with st.expander("Preview loaded jobs"):
            for i, job in enumerate(st.session_state.jobs_data[:5]):
                st.markdown(
                    f"**{i+1}. {job.get('title', 'N/A')}** at "
                    f"{job.get('company', 'N/A')} — {job.get('location', 'N/A')}"
                )
            if len(st.session_state.jobs_data) > 5:
                st.caption(f"... and {len(st.session_state.jobs_data) - 5} more")


def render_matching():
    """Render the matching and results section."""
    st.markdown("### 🎯 Step 3: Match & Analyze")

    if not st.session_state.resume_text:
        st.info("⬆️ Upload your resume first.")
        return
    if not st.session_state.jobs_data:
        st.info("⬆️ Search or upload jobs first.")
        return
    if not st.session_state.api_key_set:
        st.info("🔑 Add your Anthropic API key in the sidebar.")
        return

    if st.button("🚀 Run AI Matching", type="primary", use_container_width=True):
        with st.spinner("🧠 Analyzing matches with AI... This may take a moment."):
            try:
                from src.matcher import match_jobs

                results = match_jobs(
                    st.session_state.resume_text,
                    st.session_state.resume_profile,
                    st.session_state.jobs_data,
                )
                st.session_state.match_results = results
            except Exception as e:
                st.error(f"Matching error: {e}")
                return

    if st.session_state.match_results:
        results = st.session_state.match_results

        # Summary metrics
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_matches = sum(1 for s in scores if s >= 75)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-number">{len(results)}</div>'
                f'<div class="metric-label">Jobs Analyzed</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-number">{avg_score:.0f}%</div>'
                f'<div class="metric-label">Avg Match</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-number">{high_matches}</div>'
                f'<div class="metric-label">Strong Matches</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            top_score = max(scores) if scores else 0
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-number">{top_score:.0f}%</div>'
                f'<div class="metric-label">Best Match</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Sort options
        sort_by = st.selectbox("Sort by", ["Match Score (High → Low)", "Company A-Z"])
        if sort_by == "Match Score (High → Low)":
            results = sorted(results, key=lambda x: x["score"], reverse=True)
        else:
            results = sorted(results, key=lambda x: x.get("company", ""))

        # Render each result
        for i, result in enumerate(results):
            render_match_card(result, i)


def render_match_card(result: dict, index: int):
    """Render a single match result card."""
    score = result["score"]
    if score >= 75:
        badge_class = "score-high"
    elif score >= 50:
        badge_class = "score-mid"
    else:
        badge_class = "score-low"

    with st.container():
        st.markdown(
            f'<div class="match-card">'
            f'<span class="score-badge {badge_class}">{score:.0f}% Match</span>'
            f'&nbsp;&nbsp;<strong style="font-size:1.1rem">{result.get("title", "N/A")}</strong>'
            f'&nbsp;at&nbsp;<strong>{result.get("company", "N/A")}</strong>'
            f'&nbsp;—&nbsp;{result.get("location", "")}'
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander(f"📋 Details — {result.get('title', 'Job')}"):
            detail_col, action_col = st.columns([2, 1])

            with detail_col:
                # Matching skills
                if result.get("matching_skills"):
                    st.markdown("**✅ Matching Skills:**")
                    skills_html = " ".join(
                        f'<span class="skill-tag skill-match">{s}</span>'
                        for s in result["matching_skills"]
                    )
                    st.markdown(skills_html, unsafe_allow_html=True)

                # Skill gaps
                if result.get("skill_gaps"):
                    st.markdown("**🔴 Skill Gaps:**")
                    gaps_html = " ".join(
                        f'<span class="skill-tag skill-gap">{s}</span>'
                        for s in result["skill_gaps"]
                    )
                    st.markdown(gaps_html, unsafe_allow_html=True)

                # AI reasoning
                if result.get("reasoning"):
                    st.markdown("**🧠 AI Analysis:**")
                    st.markdown(result["reasoning"])

            with action_col:
                if result.get("url"):
                    st.link_button("🔗 View Posting", result["url"])

                # Cover letter generation
                key = f"cover_{index}"
                if st.button("✍️ Generate Cover Letter", key=f"btn_{key}"):
                    with st.spinner("Writing cover letter..."):
                        try:
                            from src.cover_letter import generate_cover_letter

                            letter = generate_cover_letter(
                                st.session_state.resume_text,
                                st.session_state.resume_profile,
                                result,
                            )
                            st.session_state.cover_letters[key] = letter
                        except Exception as e:
                            st.error(f"Error: {e}")

                if key in st.session_state.cover_letters:
                    st.markdown("---")
                    st.markdown(st.session_state.cover_letters[key])
                    st.download_button(
                        "📥 Download Letter",
                        st.session_state.cover_letters[key],
                        file_name=f"cover_letter_{result.get('company', 'job')}.txt",
                        key=f"dl_{key}",
                    )


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Header
    st.markdown('<div class="main-header">🎯 RAG Job Search Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        "Upload your resume → Search or upload jobs → Get AI-powered match scores, "
        "skill gap analysis, and tailored cover letters"
        "</div>",
        unsafe_allow_html=True,
    )

    # Check API key
    if not st.session_state.api_key_set:
        st.warning(
            "👈 **Add your Anthropic API key in the sidebar to get started.** "
            "Get one free at [console.anthropic.com](https://console.anthropic.com)"
        )

    render_resume_upload()
    st.markdown("---")
    render_job_search()
    st.markdown("---")
    render_matching()


if __name__ == "__main__":
    main()
