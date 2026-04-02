"""
Resume Parser - Extracts raw text from uploaded resume files (PDF or TXT).
"""

import io

from pypdf import PdfReader


def parse_resume(uploaded_file) -> str:
    """
    Extract text from an uploaded resume file.

    Args:
        uploaded_file: Streamlit UploadedFile object (PDF or TXT)

    Returns:
        Extracted text as a string
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return _parse_pdf(uploaded_file)
    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _parse_pdf(uploaded_file) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "Could not extract text from PDF. "
            "The file may be scanned/image-based. "
            "Try uploading a text-based PDF or a .txt file instead."
        )

    return full_text


def parse_resume_from_path(file_path: str) -> str:
    """
    Parse a resume from a file path (for CLI / testing usage).

    Args:
        file_path: Path to the resume file

    Returns:
        Extracted text
    """
    path_lower = file_path.lower()

    if path_lower.endswith(".pdf"):
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
        return "\n\n".join(text_parts)

    elif path_lower.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {file_path}")
