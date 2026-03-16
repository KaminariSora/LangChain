import json
from typing import TypedDict, List, Literal
from model_config import llm_model

class DecisionResponse(TypedDict):
    action: Literal["QUARANTINE", "BLOCK_IP", "MONITOR", "BYPASS"]
    target: str
    reasoning: str
    priority: int # คิดว่าจะใช้ 1-5 (1 คือแรงสุด)

class DecisionAgent:
    def __init__(self):
        self.llm = llm_model
        self.structured_llm = llm_model.with_structured_output(DecisionResponse)

    def decide(self, state: dict):
        print("--- [02 Decision Agent] กำลังตัดสินใจ Action ---")

        analysis = state.get("analysis_result", {})

        print(analysis)

        prompt = f"""
        คุณคือ Security Decision Maker
        ข้อมูลการวิเคราะห์จาก Analyst: {json.dumps(analysis)}
        
        กฎการตัดสินใจ (Policy):
        - ถ้า Severity คือ CRITICAL หรือ HIGH ให้เลือก QUARANTINE เสมอ
        - ถ้า Severity คือ MEDIUM และ Confidence > 80% ให้เลือก BLOCK_IP
        - นอกเหนือจากนั้นให้ MONITOR
        
        ภารกิจ: เลือก Action ที่เหมาะสมที่สุดและอธิบายเหตุผล
        """
        
        decision = self.structured_llm.invoke(prompt)
        
        return {"decision": decision}