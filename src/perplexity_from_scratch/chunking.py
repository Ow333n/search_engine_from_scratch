
from pdf_loader import load_pdfs


def chunk_document(doc, chunk_size=100, overlap=20):
    words = doc["text"].split()
    chunks = []

    if not words:
        return chunks

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        chunk = {
            "id": f"{doc['id']}_C{len(chunks) + 1}",
            "title": doc["title"],
            "source": doc["source"],
            "page": doc["page"],
            "text": chunk_text
        }

        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def prepare_documents(pages):
    documents = []
    skipped_references = 0
    skipped_covers = 0

    for page in pages:
        text = page["text"].strip()
        words = text.split()

        if not words:
            continue

        # 跳过参考文献页面
        if text.upper().startswith("REFERENCES"):
            skipped_references += 1
            continue

        # 跳过较短的封面页
        if page["page"] == 1 and len(words) < 35:
            skipped_covers += 1
            continue

        chunks = chunk_document(page)
        documents.extend(chunks)

    print("Skipped reference pages:", skipped_references)
    print("Skipped cover pages:", skipped_covers)

    return documents


if __name__ == "__main__":
    pages = load_pdfs("data/pdfs")
    documents = prepare_documents(pages)

    print("Original pages:", len(pages))
    print("Total chunks:", len(documents))

    if documents:
        print("\nFirst chunk:")
        print(documents[0])
