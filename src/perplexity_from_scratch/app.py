
import os
import gradio as gr

from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

from pdf_loader import load_pdfs
from chunking import prepare_documents
from inverted_index import build_inverted_index
from rag import rag_search


# 1. 加载 PDF
print("Loading PDFs...")

pages = load_pdfs("data/pdfs")
documents = prepare_documents(pages)

print("Total chunks:", len(documents))


# 2. 建立索引
index = build_inverted_index(documents)
doc_map = {doc["id"]: doc for doc in documents}


# 3. Embedding Model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [doc["text"] for doc in documents]
doc_embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# 4. Cross-Encoder
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)


# 5. DeepSeek
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


# 6. 问答
def ask(question):

    if not question.strip():
        return "Please enter a question.", ""

    result = rag_search(
        question,
        index,
        doc_map,
        documents,
        doc_embeddings,
        model,
        client,
        topk=3,
        reranker=reranker
    )

    status = result["status"]

    # 资料不足：不展示任何引用
    if status == "insufficient":
        return (
            "The retrieved lecture excerpts do not contain "
            "enough information to answer this question.",
            "No supporting lecture sources found."
        )

    # 模型输出异常：不冒险展示未经验证的答案
    if status != "answered":
        return (
            result["answer"],
            "No validated citations available."
        )

    answer = result["answer"]
    sources = []

    # 使用结构化 citations，不再用正则猜测
    for i in result["citations"]:

        item = result["results"][i - 1]
        doc = doc_map[item["id"]]

        source = doc["source"]
        page = doc["page"]
        text = doc["text"][:500]

        source_text = (
            f"**[{i}] {source} — PDF Page {page}**\n\n"
            f"> {text}"
        )

        sources.append(source_text)

    return answer, "\n\n---\n\n".join(sources)


# 7. Gradio
with gr.Blocks() as demo:

    gr.Markdown("# KCL Course AI")

    gr.Markdown(
        "Lecture-grounded Q&A powered by "
        "BM25 + Dense Retrieval + RRF + "
        "Cross-Encoder + DeepSeek"
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What is Little's Law?",
        lines=2
    )

    button = gr.Button("Search", variant="primary")

    gr.Markdown("## AI Answer")
    answer_box = gr.Markdown()

    gr.Markdown("## Cited Lecture Sources")
    sources_box = gr.Markdown()

    button.click(
        fn=ask,
        inputs=question,
        outputs=[answer_box, sources_box]
    )

    question.submit(
        fn=ask,
        inputs=question,
        outputs=[answer_box, sources_box]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
