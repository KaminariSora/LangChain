import json
from pydantic import BaseModel
from typing import Optional
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

class AgentResponse(BaseModel):
    intent: str                # ผู้ใช้ต้องการอะไร
    tool_used: Optional[str]   # เรียก tool อะไร
    data: Optional[str]        # ผลลัพธ์จาก tool
    final_answer: str 

llm = ChatOllama(
    model="gpt-oss:latest",
    temperature=0.8,
    # base_url="http://localhost:11434"  # default ไม่ต้องใส่
)

system_prompt = """
    You are a helpful assistant.

    Always respond in JSON that matches this schema:

    {
    "intent": string,
    "tool_used": string | null,
    "data": string | null,
    "final_answer": string
    }

    Be concise and accurate.
    """

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@tool
def say_hello() -> str:
    """answer user's question when user say "Helloworld." """
    return f"Hi there, how are you today?"

agent = create_agent(
    llm, 
    tools=[search, get_weather, say_hello], 
    system_prompt=system_prompt
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

raw_output = result["messages"][-1].content
parsed = AgentResponse.model_validate(json.loads(raw_output))

print(parsed)