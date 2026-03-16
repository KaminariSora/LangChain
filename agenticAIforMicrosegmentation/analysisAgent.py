import json
from typing import TypedDict, List
from model_config import llm_model

class AnalysisResponse(TypedDict):
    severity: str
    blast_radius: List[str]
    confidence: int

class AgentState(TypedDict):
    raw_alert: dict
    topology: dict
    analysis_result: dict  # ผลลัพธ์จาก Agent 1
    decision: dict         # ผลลัพธ์จาก Agent 2
    execution_log: list    # ผลลัพธ์จาก Agent 3

class AnalystAgent: 
    def __init__(self):
        self.llm = llm_model
        self.structured_llm = llm_model.with_structured_output(AnalysisResponse)

    def analyze(self, state: AgentState):
        print("--- [01 Analyst Agent] กำลังวิเคราะห์... ---")
        
        prompt = f"""คุณคือ Security Analyst. 
        วิเคราะห์ Alert นี้: {json.dumps(state['raw_alert'])}
        โดยอ้างอิงจาก Network Topology: {json.dumps(state['topology'])}
        ประเมินความรุนแรงและผลกระทบ (Blast Radius) มาให้ละเอียด"""
        
        analysis = self.structured_llm.invoke(prompt)
        
        return {"analysis_result": analysis}