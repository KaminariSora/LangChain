from vectorStore import retriever
from rag_chain import rag_chain
import time

def build_context(docs):
    return "\n\n".join(
        f"""
สินค้า: {doc.page_content}
ราคา: {doc.metadata.get('price')} บาท
หมวด: {doc.metadata.get('category')}
"""
        for doc in docs
    )

while True:
    question = input("input: ")

    if question.lower() in ["exit", "quit"]:
        break

    # retrieve
    print("searching information...")
    total_start = time.time()
    # ---- RETRIEVE ----
    t1 = time.time()
    docs = retriever.invoke(question)
    retrieve_time = time.time() - t1

    if not docs:
        print("ไม่พบสินค้าที่เกี่ยวข้อง")
        continue

    # build context
    t2 = time.time()
    context = build_context(docs)
    context_time = time.time() - t2

    # RAG
    t3 = time.time()
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })
    llm_time = time.time() - t3
    total_time = time.time() - total_start

    print("\nตอบ:", answer)
    print(f"""
    ⏱️ เวลา:
    - Retrieve : {retrieve_time:.2f} วินาที
    - Context  : {context_time:.2f} วินาที
    - LLM      : {llm_time:.2f} วินาที
    -------------------------
    - รวม      : {total_time:.2f} วินาที
    """)
