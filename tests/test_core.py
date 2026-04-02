"""
Tests for RAG Job Search Agent core modules.
Run with: pytest tests/ -v
"""

import pytest
import os
import json


# --- Resume Parser Tests ---

class TestResumeParser:
    def test_parse_txt_file(self, tmp_path):
        """Test parsing a plain text resume."""
        from src.resume_parser import parse_resume_from_path

        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("Sanjay Kandimalla\nData Analyst\nSkills: Python, SQL")

        result = parse_resume_from_path(str(txt_file))
        assert "Sanjay" in result
        assert "Python" in result

    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        from src.resume_parser import parse_resume_from_path

        with pytest.raises(ValueError, match="Unsupported"):
            parse_resume_from_path("resume.docx")


# --- Job Fetcher Tests ---

class TestJobFetcher:
    def test_parse_csv(self, tmp_path):
        """Test parsing jobs from CSV."""
        from src.job_fetcher import parse_jobs_csv
        import io

        csv_content = (
            "title,company,description,location\n"
            '"Data Analyst","Acme","Need SQL and Python skills","Dallas, TX"\n'
            '"ML Engineer","TechCo","Looking for ML experience","Remote"\n'
        )

        class FakeFile:
            name = "jobs.csv"
            def read(self):
                return csv_content.encode("utf-8")

        jobs = parse_jobs_csv(FakeFile())
        assert len(jobs) == 2
        assert jobs[0]["title"] == "Data Analyst"
        assert jobs[1]["company"] == "TechCo"

    def test_parse_csv_empty(self, tmp_path):
        """Test that empty CSV raises ValueError."""
        from src.job_fetcher import parse_jobs_csv

        class FakeFile:
            name = "empty.csv"
            def read(self):
                return b"title,company,description\n"

        with pytest.raises(ValueError, match="No valid jobs"):
            parse_jobs_csv(FakeFile())

    def test_create_sample_csv(self):
        """Test sample CSV generation."""
        from src.job_fetcher import create_sample_csv

        csv = create_sample_csv()
        assert "title" in csv
        assert "description" in csv

    def test_api_without_key(self):
        """Test that API search fails gracefully without key."""
        from src.job_fetcher import search_jobs_api

        os.environ.pop("JSEARCH_API_KEY", None)
        with pytest.raises(ValueError, match="JSEARCH_API_KEY"):
            search_jobs_api("Data Analyst")


# --- Profile Extractor Tests ---

class TestProfileExtractor:
    def test_missing_api_key(self):
        """Test that extraction fails without API key."""
        from src.profile_extractor import extract_profile

        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            extract_profile("Some resume text")


# --- Matcher Tests ---

class TestMatcher:
    def test_build_job_text(self):
        """Test job text construction for embedding."""
        from src.matcher import _build_job_text

        job = {
            "title": "Data Analyst",
            "company": "Acme",
            "location": "Dallas",
            "description": "SQL and Python required",
        }
        text = _build_job_text(job)
        assert "Data Analyst" in text
        assert "SQL and Python" in text

    def test_job_id_consistency(self):
        """Test that job IDs are stable."""
        from src.matcher import _job_id

        job = {"title": "Analyst", "company": "Acme"}
        id1 = _job_id(job, 0)
        id2 = _job_id(job, 0)
        assert id1 == id2

    def test_job_id_uniqueness(self):
        """Test that different jobs get different IDs."""
        from src.matcher import _job_id

        job1 = {"title": "Analyst", "company": "Acme"}
        job2 = {"title": "Engineer", "company": "Acme"}
        assert _job_id(job1, 0) != _job_id(job2, 1)

    def test_empty_jobs(self):
        """Test matcher with empty job list."""
        from src.matcher import match_jobs

        results = match_jobs("resume text", None, [])
        assert results == []
