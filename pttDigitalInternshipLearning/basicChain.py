from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama

# input
userInput = input(str("user: "))

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        # Prompt = ข้อความทั้งหมด (system + ทุก human + ทุก assistant ก่อนหน้า) ที่ส่งไปให้โมเดลจริง ๆ
        # System Message = บอก AI ว่า "แกเป็นใคร ต้องทำตัวยังไง กฎคืออะไร" → ตั้งไว้ตอนแรก ใช้ตลอด
        # Human Message = ข้อความที่ "คนจริง" ถาม/สั่งงาน → เปลี่ยนทุกครั้งที่คุยรอบใหม่  
        ("system", "คุณเป็น {expertise}"),
        ("human", userInput),
    ]
)

# ใช้ Ollama (local model)
llm = ChatOllama(
    model="bge-m3:latest",
    temperature=0.8,
    # base_url="http://localhost:11434"  # default ไม่ต้องใส่
)

# parser
Json_parser = JsonOutputParser()
Comma_parser = CommaSeparatedListOutputParser()

# chain
chain = prompt | llm | Comma_parser

# --------------------
# เรียกใช้ model ตรง ๆ
# --------------------
response = llm.invoke(userInput)
print("Bot response:", response.content)

# --------------------
# เรียกใช้ผ่าน Chain
# --------------------
response_chain = chain.invoke({
    "expertise": "ที่ปรึกษาที่ดีที่สุด",
    "topic": "ปรึกษาปัญหาชีวิต",
})

print("Chain response:", response_chain)
