import fitz  # PyMuPDF — no poppler needed
import trafilatura


def extract_text_from_url(url: str) -> str:
    """
    Downloads a webpage and extracts only the main article text.
    Works for Bengali news sites like Prothom Alo, Anandabazar, etc.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not download content from URL: {url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text or len(text.strip()) < 50:
        raise ValueError("No readable text found at this URL. Try a different link.")

    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    PDF reader using PyMuPDF (no poppler needed).
    Supports text-based/digital PDFs only.
    """
    full_text_parts = []

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    print(f"[utils] PDF has {total_pages} pages.")

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if text and len(text) > 30:
            full_text_parts.append(text)
            print(f"[utils] Page {i+1}: digital text ✓")
        else:
            print(f"[utils] Page {i+1}: no extractable text (likely scanned/image page), skipping")

    doc.close()

    if not full_text_parts:
        raise ValueError(
            "Could not extract any text from this PDF. "
            "The PDF may be fully image-based. "
            "Please upload a text-based/digital PDF."
        )

    return "\n\n".join(full_text_parts)


def split_into_batches(text: str, chars_per_batch: int = 8000) -> list:
    """
    Splits large text into smaller chunks (~10 pages each)
    so we can send them one-by-one to the Gemini API for free tier safety.
    """
    words = text.split()
    batches = []
    current_batch = []
    current_length = 0

    for word in words:
        current_batch.append(word)
        current_length += len(word) + 1

        if current_length >= chars_per_batch:
            batches.append(" ".join(current_batch))
            current_batch = []
            current_length = 0

    if current_batch:
        batches.append(" ".join(current_batch))

    print(f"[utils] Text split into {len(batches)} batches.")
    return batches
