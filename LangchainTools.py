import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# ตั้งค่าตัวแปรสิ่งแวดล้อม
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Working/C_work/Coding/LangChain/.env/service_account.json"

# ฟังก์ชันจริงที่ใช้คำนวณ
def multiply_func(a: float, b: float) -> float:
    return a * b

def division_func(a: float, b: float) -> float:
    return a / b

# สร้าง Tool โดยใช้ @tool
@tool
def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return multiply_func(a, b)

@tool
def division(a: float, b: float) -> float:
    """Division a and b."""
    return division_func(a, b)

# โหลดโมเดล Gemini
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8
)

# ผูกฟังก์ชันเครื่องมือเข้ากับโมเดล
llm_with_tools = llm.bind_tools([multiply])

# ทดสอบ prompt ธรรมดา
result = llm_with_tools.invoke("What is 2 multiplied by 5?")
print(f"Without tools: '{result.content}'")
print("Tool calls:", result.tool_calls)

# ทดสอบคำถามที่ต้องใช้ tool
normal_result = llm.invoke("What is 2 multiplied by 5?")
print(normal_result.content)

result_tools = llm_with_tools.invoke("What is 2 multiplied by 5?")

if result_tools.tool_calls:
    call = result_tools.tool_calls[0]
    a = float(call['args']['a'])
    b = float(call['args']['b'])
    answer = multiply_func(a, b)
    print(f"Model asked to multiply {a} and {b}, result = {answer}")
else:
    print(result_tools.content)
   
result_tools = llm_with_tools.invoke("What is 2 multiplied by 5?")

if result_tools.tool_calls:
    call = result_tools.tool_calls[0]
    a = float(call['args']['a'])
    b = float(call['args']['b'])
    answer = division_func(a, b)
    print(f"Model asked to division {a} and {b}, result = {answer}")
else:
    print(result_tools.content)
