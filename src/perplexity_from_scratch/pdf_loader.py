from pathlib import Path
import pymupdf


def load_pdfs(folder):

    documents = []

    pdf_files = sorted(Path(folder).glob('*.pdf'))

    for pdf_path in pdf_files:

        with pymupdf.open(pdf_path) as pdf:

            for page_index, page in enumerate(pdf):
                text = page.get_text('text', sort = True).strip()
                if not text:
                    continue
                doc = {
                    'id': f'D{len(documents) + 1}',
                    'title': pdf_path.stem,
                    'source':pdf_path.name,
                    'page': page_index + 1,
                    'text': text
                }

                documents.append(doc)
    return documents

if __name__ == '__main__':
    documents = load_pdfs('data/pdfs')
    print("Total documents:", len(documents))

    print("\nFirst document:")
    print("ID:", documents[0]["id"])
    print("Source:", documents[0]["source"])
    print("Page:", documents[0]["page"])
    print("Text:", documents[0]["text"][:500])
    context = build_context(results[:2], doc_map)

    print("\n=== CONTEXT FOR DEEPSEEK ===")
    print(context)