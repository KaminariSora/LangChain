from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_ollama import ChatOllama

from detect import event

# print(event)

event_log = '''
type=USER_CMD msg=audit(1768751663.263:688): pid=2870 uid=1000 exe="/usr/bin/sudo" cmd="/usr/sbin/cron" tty=/dev/tty1
'''

# input
userInput = "ตรวจจับความผิดปกติและบอก policy ที่แนะนำมา"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        # Prompt = ข้อความทั้งหมด (system + ทุก human + ทุก assistant ก่อนหน้า) ที่ส่งไปให้โมเดลจริง ๆ
        # System Message = บอก AI ว่า "แกเป็นใคร ต้องทำตัวยังไง กฎคืออะไร" → ตั้งไว้ตอนแรก ใช้ตลอด
        # Human Message = ข้อความที่ "คนจริง" ถาม/สั่งงาน → เปลี่ยนทุกครั้งที่คุยรอบใหม่  
        ("system", "คุณเป็น {expertise} จาก {log} โดยมี {extract_event} เป็นพื้นฐานในการดำเนินการ"),
        ("human", userInput),
    ]
)

# ใช้ Ollama (local model)
llm = ChatOllama(
    model="thewindmom/llama3-med42-8b",
    temperature=0.8,
    # base_url="http://localhost:11434"  # default ไม่ต้องใส่
)

# parser
Json_parser = JsonOutputParser()
Comma_parser = CommaSeparatedListOutputParser()

# chain
chain = prompt | llm | Json_parser

# --------------------
# เรียกใช้ model ตรง ๆ
# --------------------
response = llm.invoke(userInput)
print("Bot response:", response.content)

# --------------------
# เรียกใช้ผ่าน Chain
# --------------------
response_chain = chain.invoke({
    "expertise": "Agent ที่คอยตรวจจับความผิดปกติของระบบ microsegmentation",
    "log": "{event_log}",
    "extract_event": "{event}"
})

print("Chain response:", response_chain)
