import os, vertexai
from vertexai.generative_models import GenerativeModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Working\C_work\Coding\LangChain\.env\service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "gcp-pttdigital-lab"
os.environ["VERTEX_LOCATION"] = "us-central1"

vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["VERTEX_LOCATION"])
resp = GenerativeModel("gemini-2.0-flash").generate_content("พิมพ์ OK")
print(resp.text)