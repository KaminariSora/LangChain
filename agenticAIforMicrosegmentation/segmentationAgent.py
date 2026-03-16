import json
from typing import TypedDict, List
from model_config import llm_model

# 1. โครงสร้างรายงานผลการปฏิบัติงาน
class ExecutionReport(TypedDict):
    status: str          
    action_taken: str    
    details: str         
    next_step: str      

class ExecutionAgent:
    def __init__(self):
        self.llm = llm_model
        self.structured_llm = llm_model.with_structured_output(ExecutionReport)

    def execute(self, state: dict):
        print("--- [03 Execution Agent] กำลังดำเนินการตามคำสั่ง ---")
        
        decision = state.get("decision", {})
        target_assets = state.get("analysis_result", {}).get("blast_radius", [])
        
        prompt = f"""
        คุณคือ Security Automation Engineer
        ภารกิจ: ดำเนินการตามการตัดสินใจดังนี้:
        - Action: {decision.get('action')}
        - Reason: {decision.get('reasoning')}
        - Target Assets: {target_assets}

        หาก Action คือ QUARANTINE: ให้จำลองการตัดการเชื่อมต่อของ Assets ทั้งหมด
        หาก Action คือ BLOCK_IP: ให้จำลองการเพิ่ม IP ใน Blacklist ของ Firewall
        
        ตอบกลับเป็นรายงานผลการปฏิบัติงานในรูปแบบ JSON
        """
        
        execution_result = self.structured_llm.invoke(prompt)
        
        return {"execution_log": [execution_result]}