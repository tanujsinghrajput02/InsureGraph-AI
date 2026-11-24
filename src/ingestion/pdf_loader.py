
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> list[Document]:
    """
    Extract text from a PDF and return LangChain Documents.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    try:
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": path.name,
                                "page": page_number
                            }
                        )
                    )

    except Exception as error:
        raise RuntimeError(f"Could not read PDF: {error}") from error

    if not documents:
        raise ValueError(
            "No readable text found in this PDF. "
            "It may be empty or scanned."
        )

    return documents
