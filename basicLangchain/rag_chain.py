from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(".env/.env")
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณเป็นแอดมินร้านค้า ตอบจากข้อมูลด้านล่างเท่านั้น ห้ามเดาข้อมูลเพิ่ม\n\n{context}"
    ),
    ("human", "{question}")
])

# llm = ChatOllama(
#     # model="gpt-oss:latest",
#     model="thewindmom/llama3-med42-8b:latest",
#     temperature=0.3
# )

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.3
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # เร็ว + ถูก
    temperature=0.3,
    api_key=google_api_key
)

rag_chain = prompt | llm | StrOutputParser()