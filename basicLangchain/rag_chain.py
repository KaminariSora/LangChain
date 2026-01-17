from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณเป็นแอดมินร้านค้า ตอบจากข้อมูลด้านล่างเท่านั้น ห้ามเดาข้อมูลเพิ่ม\n\n{context}"
    ),
    ("human", "{question}")
])

llm = ChatOllama(
    model="gpt-oss:latest",
    temperature=0.3
)

rag_chain = prompt | llm | StrOutputParser()