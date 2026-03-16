from analysisAgent import AnalystAgent
from decisionAgent import DecisionAgent
from segmentationAgent import ExecutionAgent
import json

with open('./agenticAIforMicrosegmentation/Data/raw_alert.txt', 'r') as file:
    raw_alert = file.read()

with open('./agenticAIforMicrosegmentation/Data/topology.json', 'r') as file:
    topology = json.load(file)


if __name__ == "__main__":
    mock_data = {
        "raw_alert": raw_alert,
        "topology": topology
    }
    
    analyst = AnalystAgent()
    analyst_result = analyst.analyze(mock_data)
    
    print("\n[Analyst Result]:", analyst_result["analysis_result"])

    decision = DecisionAgent()
    decision_result = decision.decide(analyst_result)

    print("\n[Decision Result]:", decision_result["decision"])

    segmentation = ExecutionAgent()
    segmentation_result = segmentation.execute(decision_result)

    print("\n[Execute Result]:", segmentation_result["execution_log"])
    print("\nProgram Ended..")