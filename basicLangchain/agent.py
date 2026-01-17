from vectorStore import retriever
from rag_chain import rag_chain

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
    docs = retriever.invoke(question)

    if not docs:
        print("ไม่พบสินค้าที่เกี่ยวข้อง")
        continue

    # build context
    context = build_context(docs)

    # RAG
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nตอบ:", answer)
