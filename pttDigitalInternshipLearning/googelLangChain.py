import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Working/C_work/Coding/LangChain/.env/service_account.json"

from langchain.chat_models import init_chat_model

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","คุณเป็น {expertise}"),
        ("human","แนะนำเกี่ยวกับ {topic} จำนวน {amount} รายการ")
        # MessagesPlaceholder(variable_name="messages")
    ]
)

# model
llm = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_vertexai",
    project="gcp-pttdigital-lab",
    location="us-central1",
    temperature=0.8,
)

# parser
parser = CommaSeparatedListOutputParser()

# chain
chain = prompt | llm | parser
# print(chain)

# เรียกใช้งาน model
response = llm.invoke("นายกรัฐมนตรีของไทยคนล่าสุดคือใคร")
print("Bot response: ",response.content)

# เรียกใช้งาน Chain
# response_chain = chain.invoke({"expertise":"เชฟ","topic":"เมนู","amount":5})
# print("Chain response: ",response_chain)