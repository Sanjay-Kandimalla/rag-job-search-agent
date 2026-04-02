
**AI-powered job matching and cover letter generator using RAG (Retrieval-Augmented Generation)**

Upload your resume → Search or upload jobs → Get AI-powered match scores, skill gap analysis, and tailored cover letters.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B.svg)](https://rag-job-search-agent-6n5abkmuemkjkfrwdydjrz.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🔗 **[Live Demo](https://rag-job-search-agent-6n5abkmuemkjkfrwdydjrz.streamlit.app/)**

---

## 🚀 What It Does

Most job seekers apply to hundreds of postings blindly. This tool uses **semantic search** and **LLM reasoning** to actually understand how well your background matches each job — then explains *why* and writes you a tailored cover letter.

| Feature | How It Works |
|---|---|
| **Resume Parsing** | Extracts text from PDF/TXT, then uses Claude to identify skills, experience, education, and target roles |
| **Job Search** | Live search via JSearch API (500+ free requests/month) or upload your own CSV of job postings |
| **RAG Matching** | Embeds resume + jobs using Sentence-BERT → stores in ChromaDB → retrieves top semantic matches → Claude scores and analyzes each one |
| **Skill Gap Analysis** | For each job: which of your skills match, which are missing, and a plain-English explanation of your fit |
| **Cover Letter Gen** | One-click tailored cover letters that reference your specific skills and the job's requirements |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│   Resume Upload │ Job Search │ Match Results │ Cover Letters    │
└────────┬────────┴─────┬──────┴──────┬────────┴───────┬──────────┘
│              │             │                │
▼              ▼             ▼                ▼
┌─────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐
│   Resume    │  │    Job     │  │  RAG Match   │  │Cover Letter  │
│   Parser    │  │  Fetcher   │  │   Engine     │  │  Generator   │
│  (pypdf)    │  │(JSearch/CSV│  │              │  │  (Claude)    │
└──────┬──────┘  └─────┬──────┘  │  ┌────────┐  │  └──────────────┘
│               │         │  │Sentence│  │
▼               │         │  │ BERT   │  │
┌─────────────┐        │         │  │Embedder│  │
│  Profile    │        │         │  └───┬────┘  │
│ Extractor   │        │         │      │       │
│  (Claude)   │        │         │  ┌───▼────┐  │
└─────────────┘        │         │  │ChromaDB│  │
│         │  │ Vector │  │
│         │  │ Store  │  │
│         │  └───┬────┘  │
│         │      │       │
│         │  ┌───▼────┐  │
└─────────┤  │ Claude │  │
│  │Analyzer│  │
│  └────────┘  │
└──────────────┘

### RAG Pipeline (core matching logic)
Resume Text ──► Sentence-BERT ──► Resume Embedding (384-dim vector)
│
│ cosine similarity query
▼
Job Descriptions ──► Sentence-BERT ──► ChromaDB Vector Store
│
│ top-k retrieval
▼
Matched Jobs + Resume
│
│ detailed analysis prompt
▼
Claude Sonnet 4
│
▼
Match Score + Skill Gaps
+ Reasoning + Cover Letter

**Why RAG instead of just sending everything to an LLM?**

- **Cost**: Embedding 50 jobs costs ~$0.00. Sending all 50 full job descriptions to Claude would cost ~$0.50+ per query.
- **Speed**: Vector similarity search returns in milliseconds. Analyzing all jobs with Claude would take minutes.
- **Accuracy**: Semantic embeddings catch matches that keyword search misses ("ML" ↔ "machine learning", "dashboards" ↔ "data visualization"). Claude then adds the nuanced reasoning layer only for relevant matches.

---

## 📦 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | Streamlit | Fast prototyping, interactive widgets, easy deployment |
| **LLM** | Anthropic Claude (Sonnet 4) | Best reasoning for match analysis and writing |
| **Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) | Fast, free, runs locally — no API costs for embeddings |
| **Vector DB** | ChromaDB | Lightweight, in-memory, zero config — ideal for this scale |
| **PDF Parsing** | pypdf | Pure Python, no system dependencies |
| **Job Data** | JSearch API + CSV | Real jobs from API, or bring your own data |
| **CI/CD** | GitHub Actions | Automated linting, testing, Docker build on every push |
| **Container** | Docker | Reproducible deployment anywhere |

---

## ⚡ Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Sanjay-Kandimalla/rag-job-search-agent.git
cd rag-job-search-agent
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
# Get one at: https://console.anthropic.com
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run
```bash
streamlit run app.py
```

Open `http://localhost:8501` and you're live.

### Docker (alternative)
```bash
docker compose up --build
```

---

## 🔑 API Keys

| Key | Required? | Free Tier | Get It |
|---|---|---|---|
| **Anthropic** | ✅ Yes | Pay-as-you-go (~$0.01/match) | [console.anthropic.com](https://console.anthropic.com) |
| **JSearch** | ❌ Optional | 500 requests/month | [rapidapi.com/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) |

> **No JSearch key?** No problem — use the CSV upload tab with the included `data/sample_jobs.csv` to try the matching engine.

---

## 📁 Project Structure
rag-job-search-agent/
│
├── app.py                        # Streamlit UI — entry point
├── src/
│   ├── resume_parser.py          # PDF/TXT text extraction
│   ├── profile_extractor.py      # Claude-powered skill/experience extraction
│   ├── job_fetcher.py            # JSearch API client + CSV parser
│   ├── matcher.py                # ⭐ RAG pipeline: SBERT → ChromaDB → Claude
│   └── cover_letter.py           # Tailored cover letter generation
│
├── tests/
│   └── test_core.py              # Unit tests for all modules
│
├── data/
│   └── sample_jobs.csv           # 10 sample job postings for demo
│
├── .github/workflows/ci.yml     # CI: lint + test + Docker build
├── Dockerfile                    # Production container
├── docker-compose.yml            # Local dev with Docker
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .streamlit/config.toml        # Streamlit theme config
└── README.md                     # You are here

---

## 🧪 Testing
```bash
pip install pytest ruff
pytest tests/ -v          # Run test suite
ruff check src/ app.py    # Lint check
```

---

## 🎯 How Matching Works — In Detail

**Step 1: Resume Understanding**
Your resume PDF is parsed into raw text, then sent to Claude with a structured extraction prompt. Claude returns a JSON profile: skills, experience years, education, target job titles, and key achievements.

**Step 2: Semantic Embedding**
Both your resume and all job descriptions are converted to 384-dimensional vectors using Sentence-BERT. This captures *meaning*, not just keywords — so "built ML pipelines" matches "machine learning engineering" even though the words are different.

**Step 3: Vector Retrieval**
Job embeddings are stored in ChromaDB. Your resume embedding is used as a query — ChromaDB returns the most semantically similar jobs ranked by cosine similarity. This is the "Retrieval" in RAG.

**Step 4: LLM Analysis**
The top matches (resume + job descriptions) are sent to Claude for detailed analysis. Claude returns:
- **Match score** (0-100) with calibrated guidelines
- **Matching skills** — what you have that they want
- **Skill gaps** — what's missing and how critical it is
- **Reasoning** — plain-English explanation of fit

**Step 5: Cover Letter Generation**
For any match, one click generates a tailored cover letter that references your specific skills, addresses skill gaps honestly, and uses details from the job posting.

---

## 🛠️ Extending This Project

Some ideas if you want to fork and build on this:

- **Add more job sources**: Indeed, LinkedIn (with scraping), or Greenhouse API
- **Job alerts**: Schedule daily searches and email new high-match jobs
- **Interview prep**: Generate likely interview questions based on skill gaps
- **Multi-resume support**: Compare how different resume versions score against the same jobs
- **Analytics dashboard**: Track application history, response rates, score distributions

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Sanjay Kandimalla**
MS Applied Statistics & Data Science, UT Arlington

- [LinkedIn](https://linkedin.com/in/sanjay-kandimalla/)
- [GitHub](https://github.com/Sanjay-Kandimalla)
- [Email](mailto:sanjay.kandimalla2025@gmail.com)
