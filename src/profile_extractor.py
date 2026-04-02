"""
Profile Extractor - Uses Claude to extract a structured skills/experience
profile from raw resume text.
"""

import json
import os

from anthropic import Anthropic


EXTRACTION_PROMPT = """Analyze this resume and extract a structured profile. Return ONLY valid JSON with no markdown formatting, no backticks, no explanation.

JSON schema:
{{
    "skills": ["list of technical skills, tools, languages, frameworks"],
    "soft_skills": ["list of soft skills like communication, leadership, etc."],
    "job_titles": ["target job titles this person is suitable for"],
    "experience_years": "estimated total years of professional experience as a string",
    "education": "highest degree and field, e.g., 'MS Applied Statistics, UT Arlington'",
    "industries": ["industries this person has worked in or studied"],
    "key_achievements": ["2-3 most impressive quantifiable achievements"],
    "summary": "2-3 sentence professional summary"
}}

Resume text:
{resume_text}
"""


def extract_profile(resume_text: str) -> dict:
    """
    Use Claude to extract a structured profile from resume text.

    Args:
        resume_text: Raw text extracted from resume

    Returns:
        Dictionary with structured profile data
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(resume_text=resume_text[:6000]),
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Clean up common LLM response artifacts
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1]
    if response_text.endswith("```"):
        response_text = response_text.rsplit("```", 1)[0]
    response_text = response_text.strip()

    try:
        profile = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            profile = json.loads(response_text[start:end])
        else:
            raise ValueError(f"Could not parse profile from LLM response: {response_text[:200]}")

    # Validate expected keys
    expected_keys = ["skills", "job_titles", "experience_years", "education", "summary"]
    for key in expected_keys:
        if key not in profile:
            profile[key] = [] if key in ("skills", "job_titles") else ""

    return profile
