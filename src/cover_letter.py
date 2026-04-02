"""
Cover Letter Generator - Uses Claude to generate tailored cover letters
based on resume profile and specific job match analysis.
"""

import os
from typing import Optional
from anthropic import Anthropic


COVER_LETTER_PROMPT = """Write a professional, tailored cover letter for this job application.

CANDIDATE INFO:
{resume_summary}

EXTRACTED PROFILE:
- Skills: {skills}
- Experience: {experience}
- Education: {education}
- Key Achievements: {achievements}

JOB DETAILS:
- Title: {job_title}
- Company: {company}
- Location: {location}
- Description: {job_description}

MATCH ANALYSIS:
- Match Score: {score}%
- Matching Skills: {matching_skills}
- Skill Gaps: {skill_gaps}
- Analysis: {reasoning}

INSTRUCTIONS:
1. Write a 3-4 paragraph cover letter (250-350 words)
2. Open with enthusiasm for the specific role and company
3. Highlight 2-3 specific skills/achievements from the resume that match the job requirements
4. Address skill gaps honestly — frame them as areas of active learning or transferable skills
5. Close with a confident call to action
6. Use a professional but warm tone — not generic or robotic
7. Reference specific details from the job description to show it's tailored
8. DO NOT use clichés like "I am writing to express my interest" or "I believe I would be a great fit"

Return ONLY the cover letter text, no headers or metadata.
"""


def generate_cover_letter(
    resume_text: str,
    resume_profile: Optional[dict],
    match_result: dict,
) -> str:
    """
    Generate a tailored cover letter for a specific job match.

    Args:
        resume_text: Raw resume text
        resume_profile: Extracted profile dict
        match_result: Match result dict from the matcher

    Returns:
        Cover letter text
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    profile = resume_profile or {}

    prompt = COVER_LETTER_PROMPT.format(
        resume_summary=resume_text[:2000],
        skills=", ".join(profile.get("skills", [])[:15]),
        experience=profile.get("experience_years", "Not specified"),
        education=profile.get("education", "Not specified"),
        achievements=" | ".join(profile.get("key_achievements", [])[:3]),
        job_title=match_result.get("title", "N/A"),
        company=match_result.get("company", "N/A"),
        location=match_result.get("location", ""),
        job_description=match_result.get("description", "")[:3000],
        score=match_result.get("score", "N/A"),
        matching_skills=", ".join(match_result.get("matching_skills", [])),
        skill_gaps=", ".join(match_result.get("skill_gaps", [])),
        reasoning=match_result.get("reasoning", ""),
    )

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()
