from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI

env_path = Path("agenticAIforMicrosegmentation/.env")
load_dotenv(dotenv_path=env_path)
google_api_key = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    api_key=google_api_key
)