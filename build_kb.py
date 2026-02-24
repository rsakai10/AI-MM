
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import pdfplumber
from tqdm import tqdm
from openai import OpenAI


def read_text_from_pdf(pdf_path: str) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {page_idx + 1}]\n{page_text}")
    return "\n\n".join(text_parts)


def read_text_from_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return read_text_from_pdf(file_path)

    if suffix in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return ""


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    vectors: List[List[float]] = []
    batch_size = 64

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [item.embedding for item in resp.data]
        vectors.extend(batch_vecs)

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_kb(input_dir: str, out_dir: str, embedding_model: str) -> None:
    client = OpenAI()

    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported = [".pdf", ".txt", ".md"]
    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in supported]

    if not files:
        raise RuntimeError(f"No supported files found in {input_dir}")

    docs: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []

    print(f"Found {len(files)} files")
    for fp in tqdm(files, desc="Reading files"):
        text = read_text_from_file(str(fp))
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": len(docs),
                    "source": str(fp),
                    "chunk_index": idx,
                    "text": chunk,
                }
            )
            chunk_texts.append(chunk)

    if not chunk_texts:
        raise RuntimeError("No text chunks were generated from input files")

    embeddings = embed_texts(client, chunk_texts, model=embedding_model)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_path / "kb.index"))

    with open(out_path / "kb_meta.jsonl", "w", encoding="utf-8") as f:
        for row in docs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "embedding_model": embedding_model,
        "dimension": dim,
        "metric": "cosine_via_inner_product_after_l2_normalization",
        "num_chunks": len(docs),
    }
    with open(out_path / "kb_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("KB build completed")
    print(f"Chunks: {len(docs)}")
    print(f"Index: {out_path / 'kb.index'}")
    print(f"Metadata: {out_path / 'kb_meta.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description="Build local RAG knowledge base")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .pdf/.txt/.md files")
    parser.add_argument("--out_dir", type=str, default="./kb_store", help="Output KB folder")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    build_kb(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()