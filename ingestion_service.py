import os
import glob
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
UNSTRUCTURED_PARTITION_STRATEGY = os.getenv("UNSTRUCTURED_PARTITION_STRATEGY")

os.environ["TESSERACT_CMD"] = TESSERACT_PATH


def extract_pdf_elements(file_path: str):
    return partition_pdf(
        filename=file_path,
        strategy=UNSTRUCTURED_PARTITION_STRATEGY,
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )


def extract_from_folder(folder_path: str) -> Dict[str, list]:
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        return {}

    extracted_data = {}

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        elements = extract_pdf_elements(file_path)

        if elements:
            extracted_data[filename] = elements

    return extracted_data


def is_valid_chunk(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False

    special_ratio = sum(not c.isalnum() for c in text) / len(text)
    if special_ratio > 0.4:
        return False

    if len(set(text.strip())) < 10:
        return False

    return True


def chunk_elements(extracted_data: Dict[str, list]) -> List[Document]:
    all_docs = []

    for filename, elements in extracted_data.items():
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500,
        )

        for chunk in chunks:
            text = str(chunk)

            if not is_valid_chunk(text):
                continue

            metadata = chunk.metadata.to_dict()
            metadata["source"] = filename

            doc = Document(
                page_content=text,
                metadata=metadata
            )

            all_docs.append(doc)

    return all_docs


def store_chunks(documents: List[Document], db_path: str):
    if not documents:
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
    )

    return vector_db


def run_ingestion_pipeline(folder_path: str, db_path: str):
    extracted = extract_from_folder(folder_path)

    if not extracted:
        return None

    chunked_docs = chunk_elements(extracted)

    if not chunked_docs:
        return None

    vector_db = store_chunks(chunked_docs, db_path)

    return vector_db


def main() -> int:
    folder_path = os.getenv("PDF_FOLDER_PATH")
    db_path = os.getenv("CHROMA_DB_PATH")

    if not folder_path or not db_path:
        raise ValueError("Set PDF_FOLDER_PATH and CHROMA_DB_PATH in .env before running ingestion_service.py")

    vector_db = run_ingestion_pipeline(folder_path, db_path)
    return 0 if vector_db is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())