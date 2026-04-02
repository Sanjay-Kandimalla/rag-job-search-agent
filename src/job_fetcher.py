"""
Job Fetcher - Search for jobs via JSearch API or parse from CSV upload.
"""

import os
import csv
import io
import requests
from typing import Optional


def search_jobs_api(
    query: str,
    location: Optional[str] = None,
    num_results: int = 20,
) -> list[dict]:
    """
    Search for jobs using the JSearch API (RapidAPI).

    Args:
        query: Job search query (e.g., "Data Analyst")
        location: Optional location filter
        num_results: Number of results to fetch (max 50)

    Returns:
        List of job dictionaries
    """
    api_key = os.environ.get("JSEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "JSEARCH_API_KEY not set. Get a free key at: "
            "https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch"
        )

    url = "https://jsearch.p.rapidapi.com/search"

    search_query = query
    if location:
        search_query += f" in {location}"

    params = {
        "query": search_query,
        "page": "1",
        "num_pages": str(max(1, num_results // 10)),
        "date_posted": "month",
    }

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    jobs = []
    for item in data.get("data", [])[:num_results]:
        job = {
            "title": item.get("job_title", "N/A"),
            "company": item.get("employer_name", "N/A"),
            "location": _format_location(item),
            "description": item.get("job_description", ""),
            "url": item.get("job_apply_link") or item.get("job_google_link", ""),
            "salary": _format_salary(item),
            "date_posted": item.get("job_posted_at_datetime_utc", ""),
            "job_type": item.get("job_employment_type", ""),
            "is_remote": item.get("job_is_remote", False),
            "source": "jsearch_api",
        }
        jobs.append(job)

    return jobs


def _format_location(item: dict) -> str:
    """Format location from JSearch API response."""
    parts = []
    city = item.get("job_city")
    state = item.get("job_state")
    country = item.get("job_country")

    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if country and country != "US":
        parts.append(country)

    location = ", ".join(parts) if parts else "Not specified"

    if item.get("job_is_remote"):
        location = f"🏠 Remote — {location}" if parts else "🏠 Remote"

    return location


def _format_salary(item: dict) -> str:
    """Format salary info from JSearch API response."""
    min_sal = item.get("job_min_salary")
    max_sal = item.get("job_max_salary")
    period = item.get("job_salary_period", "")

    if min_sal and max_sal:
        return f"${min_sal:,.0f} - ${max_sal:,.0f} {period}".strip()
    elif min_sal:
        return f"${min_sal:,.0f}+ {period}".strip()
    elif max_sal:
        return f"Up to ${max_sal:,.0f} {period}".strip()
    return ""


def parse_jobs_csv(uploaded_file) -> list[dict]:
    """
    Parse job postings from an uploaded CSV file.

    Expected columns: title, company, description
    Optional columns: location, url, salary

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        List of job dictionaries
    """
    content = uploaded_file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    # Normalize column names (lowercase, strip whitespace)
    jobs = []
    for row in reader:
        normalized = {k.lower().strip(): v.strip() for k, v in row.items() if v}

        # Require at minimum a title and description
        title = (
            normalized.get("title")
            or normalized.get("job_title")
            or normalized.get("position")
        )
        description = (
            normalized.get("description")
            or normalized.get("job_description")
            or normalized.get("details")
        )

        if not title or not description:
            continue

        job = {
            "title": title,
            "company": (
                normalized.get("company")
                or normalized.get("employer")
                or normalized.get("company_name")
                or "N/A"
            ),
            "location": normalized.get("location", "Not specified"),
            "description": description,
            "url": normalized.get("url") or normalized.get("link") or "",
            "salary": normalized.get("salary") or "",
            "source": "csv_upload",
        }
        jobs.append(job)

    if not jobs:
        raise ValueError(
            "No valid jobs found in CSV. Ensure it has columns: "
            "title, company, description"
        )

    return jobs


def create_sample_csv() -> str:
    """Generate a sample CSV template for users to fill in."""
    header = "title,company,description,location,url,salary\n"
    sample = (
        '"Data Analyst","Acme Corp","Looking for a data analyst with SQL and Python '
        'experience. Must know Power BI and have experience with large datasets.",'
        '"Dallas, TX","https://example.com/job1","$65,000 - $80,000"\n'
        '"Junior Data Scientist","TechStart Inc","Entry-level data science role. '
        'Python, scikit-learn, and basic ML knowledge required.",'
        '"Remote","https://example.com/job2","$70,000 - $90,000"\n'
    )
    return header + sample
