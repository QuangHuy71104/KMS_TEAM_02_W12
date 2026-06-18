"""
Create ChromaDB from seed articles only (no Odoo required).
Run this when Odoo is not available.
"""
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = str((BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")).resolve())
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kms_collection")

from ingest_to_vector import SEED_ARTICLES, get_embedding_function


def strip_html(html):
    soup = BeautifulSoup(html or "", "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def main():
    print("Building ChromaDB from seed articles (no Odoo needed)...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []

    for art in SEED_ARTICLES:
        body_text = strip_html(art["body"])
        chunks = splitter.create_documents(
            texts=[body_text],
            metadatas=[{
                "title": art["name"],
                "workspace_dimension": art["workspace_dimension"],
                "access_role": art["access_role"],
                "tags": ", ".join(art["tags"]),
                "odoo_id": 0,
            }],
        )
        all_chunks.extend(chunks)
        print(f"  {art['code']:>4} | {art['name'][:50]:<50} | chunks={len(chunks)}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"Cleared existing {PERSIST_DIR}")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_fn = get_embedding_function()

    print(f"Saving to {PERSIST_DIR}...")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print("Done. ChromaDB ready.")


if __name__ == "__main__":
    main()
