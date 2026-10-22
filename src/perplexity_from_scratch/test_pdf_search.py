
from pdf_loader import load_pdfs
from chunking import prepare_documents
from inverted_index import build_inverted_index
from sentence_transformers import SentenceTransformer
from hybrid_search import hybrid_search
from rag import build_context


# 1. 读取所有 PDF
pages = load_pdfs("data/pdfs")

# 2. 过滤无用页面，并进行 Chunking
documents = prepare_documents(pages)

print("Original pages:", len(pages))
print("Total chunks:", len(documents))


# 3. 构建倒排索引
index = build_inverted_index(documents)

# 4. 建立 ID 到文档的映射
doc_map = {doc["id"]: doc for doc in documents}


# 5. 加载 Embedding 模型
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 6. 计算文档 Embeddings
texts = [doc["text"] for doc in documents]

doc_embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# 7. 设置测试问题
query = "What is Little's Law in queueing theory?"

# 8. Hybrid Search
results = hybrid_search(
    query,
    index,
    doc_map,
    documents,
    doc_embeddings,
    model,
    topk=5
)


# 9. 打印最终 Top 5
print("\n=== FINAL HYBRID RESULTS ===")

for rank, item in enumerate(results, start=1):
    doc = doc_map[item["id"]]

    print(f"\nRank {rank}")
    print("Source:", doc["source"])
    print("Page:", doc["page"])
    print("Score:", item["score"])
    print("Text:", doc["text"][:300])


# 10. 检查发给 DeepSeek 的 Context
top_results = results[:2]

context = build_context(top_results, doc_map)

print("\n=== CONTEXT FOR DEEPSEEK ===")
print(context)
