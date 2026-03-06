# semantic_search.py

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
# =========================
# 1. ตั้งค่า API
# =========================

client = OpenAI(
    api_key="sk-D20U9B9BlGz4kb0lcfRdfJmy4fr4t08VoLVPgCOBDhQ0i3PA",  # ใส่ key ตรงนี้
    base_url="https://api.opentyphoon.ai/v1"
)

MODEL_NAME = "typhoon-v2.5-30b-a3b-instruct"


# =========================
# 2. Dataset (records)
# =========================

records = [
"""Product: สมุดโน้ตกระดาษคราฟท์ปกแข็งลายวินเทจ
Description: สมุดปกแข็ง กระดาษคราฟท์สีน้ำตาลธรรมชาติ 160 แผ่น หนา 100 แกรม พร้อมริบบิ้นคั่นหน้า
Suitable for: ของขวัญวันเกิด, ของที่ระลึกงานแต่ง, ของแจกพนักงาน, คนชอบความวินเทจ
Style: วินเทจ, เรโทร, อบอุ่น""",

"""Product: ปากกาหมึกซึมลายไม้
Description: ปากกาหมึกซึมด้ามไม้แท้ เขียนลื่น
Suitable for: ของขวัญผู้ใหญ่, งานเกษียณ
Style: เรียบหรู""",

"""Product: เทียนหอมไขถั่วเหลืองกลิ่นลาเวนเดอร์
Description: เทียนหอมธรรมชาติ ไร้เขม่าดำ ช่วยผ่อนคลายและหลับสบาย
Suitable for: ของขวัญวันเกิด, ของขวัญขึ้นบ้านใหม่, คนชอบแต่งบ้าน
Style: มินิมอล, อบอุ่น""",

"""Product: โคมไฟตั้งโต๊ะฐานไม้
Description: โคมไฟดีไซน์นอร์ดิก ฐานทำจากไม้โอ๊คแท้ ปรับความสว่างได้ 3 ระดับ
Suitable for: อ่านหนังสือ, ตกแต่งห้องนอน, ของขวัญรับปริญญา
Style: นอร์ดิก, มินิมอล""",

"""Product: ชุดถ้วยกาแฟเซรามิกทำมือ
Description: ถ้วยกาแฟงานปั้นมือ เคลือบสีเอิร์ธโทน เอกลักษณ์ไม่ซ้ำกันในแต่ละใบ
Suitable for: คนรักกาแฟ, ของที่ระลึกงานแต่ง, ของขวัญผู้ใหญ่
Style: คราฟท์, อบอุ่น""",

"""Product: กระเป๋าผ้าแคนวาสปักลายใบไม้
Description: กระเป๋าผ้าใบหนา ทนทาน ปักลายด้วยมือ ดีไซน์รักษ์โลก
Suitable for: ของแจกพนักงาน, ของที่ระลึก, ใช้ไปเรียน
Style: ธรรมชาติ, เรียบง่าย""",

"""Product: นาฬิกาตั้งโต๊ะดิจิทัลลายไม้
Description: นาฬิกาบอกเวลาและอุณหภูมิ หน้าจอ LED ซ่อนใต้ผิวไม้
Suitable for: ตกแต่งออฟฟิศ, ของขวัญปีใหม่
Style: โมเดิร์น, มินิมอล""",

"""Product: ชุดก้านไม้หอมกระจายกลิ่น
Description: Reed Diffuser กลิ่นโอเชี่ยนเฟรช สดชื่นยาวนาน 30 วัน
Suitable for: ของขวัญงานแต่ง, ของแจกพนักงาน
Style: สดชื่น, เรียบหรู""",

"""Product: สมุดแพลนเนอร์ปกผ้าลินิน
Description: สมุดจดบันทึกรายปี กระดาษถนอมสายตา ปกหุ้มผ้าลินินสีครีม
Suitable for: คนทำงาน, นักเรียน, ของขวัญวันเกิด
Style: มูจิ, มินิมอล""",

"""Product: กล่องดนตรีไม้ไขลาน
Description: กล่องดนตรีไม้แกะสลัก เพลงคลาสสิก เสียงใสไพเราะ
Suitable for: ของขวัญวันครบรอบ, ของขวัญเด็ก
Style: วินเทจ, คลาสสิก""",

"""Product: ผ้าพันคอผ้าไหมพิมพ์ลายไทย
Description: ผ้าไหมเนื้อละเอียด พิมพ์ลายไทยประยุกต์ สีสันสดใส
Suitable for: ของขวัญชาวต่างชาติ, ของขวัญผู้ใหญ่
Style: ไทยประยุกต์, หรูหรา""",

"""Product: ชุดเครื่องเขียนโลหะสีทอง
Description: เซตปากกาและคลิปหนีบกระดาษ สีทองหรูหรา บรรจุในกล่องสวยงาม
Suitable for: ของขวัญเลื่อนตำแหน่ง, ของขวัญผู้บริหาร
Style: ลักชูรี, ทางการ""",

"""Product: กระเป๋าสตางค์หนังวัวแท้แบบพับ
Description: หนังแท้สัมผัสนุ่ม มีช่องใส่บัตร 8 ช่อง พร้อมช่องซิปใส่เหรียญ
Suitable for: ของขวัญวันเกิด, ของขวัญผู้ชาย, งานเกษียณ
Style: เรียบหรู, คลาสสิก""",

"""Product: ชุดชงชาเซรามิกสไตล์ญี่ปุ่น
Description: กาน้ำชาพร้อมถ้วย 4 ใบ ลายคลื่นทะเลญี่ปุ่น บรรจุกล่องไม้
Suitable for: ของขวัญผู้ใหญ่, คนชอบดื่มชา, ของที่ระลึก
Style: เซน, ญี่ปุ่น""",

"""Product: หมอนอิงกำมะหยี่สีเอิร์ธโทน
Description: หมอนอิงนุ่มพิเศษ ปลอกถอดซักได้ ขนาด 45x45 ซม.
Suitable for: ตกแต่งโซฟา, ของขวัญขึ้นบ้านใหม่
Style: โมเดิร์น, อบอุ่น""",

"""Product: แผ่นรองเม้าส์หนังสังเคราะห์ขนาดใหญ่
Description: แผ่นรองแบบ Desk Mat กันน้ำ ผิวสัมผัสเรียบหรู กว้าง 80 ซม.
Suitable for: จัดโต๊ะคอม, ของขวัญพนักงานออฟฟิศ
Style: มินิมอล, มืออาชีพ""",

"""Product: ร่มพับพกพาเคลือบ UV
Description: ร่มน้ำหนักเบา กันแดดและกันฝน แข็งแรงทนทานต่อลมแรง
Suitable for: ของแจกอีเวนต์, ของขวัญพนักงาน
Style: ทันสมัย""",

"""Product: ชุดปลูกแคคตัส DIY
Description: ในชุดประกอบด้วยกระถาง ดิน เมล็ดพันธุ์ และคู่มือการปลูก
Suitable for: ของขวัญเด็ก, กิจกรรมยามว่าง, คนรักต้นไม้
Style: ธรรมชาติ, น่ารัก""",

"""Product: ลำโพงไม้บลูทูธพกพา
Description: ลำโพงไร้สายดีไซน์ตัวเรือนไม้ ให้เสียงโทนอบอุ่น แบตเตอรี่อึด
Suitable for: ของขวัญวันเกิด, ตกแต่งโต๊ะทำงาน
Style: วินเทจ, ธรรมชาติ""",

"""Product: ผ้ากันเปื้อนผ้าลินินสไตล์คาเฟ่
Description: ผ้ากันเปื้อนแบบสายไขว้หลัง มีกระเป๋าหน้าใบใหญ่ เนื้อผ้าเกรดเอ
Suitable for: คนชอบทำอาหาร, เจ้าของร้านกาแฟ, ของขวัญวันแม่
Style: มินิมอล, คาเฟ่""",

"""Product: หูฟังครอบหูแบบตัดเสียงรบกวน
Description: หูฟังไร้สายระบบ ANC เบสแน่น ใส่สบายไม่บีบหู
Suitable for: คนรักเสียงเพลง, ของขวัญรับปริญญา, คนเดินทางบ่อย
Style: เทค, ทันสมัย""",

"""Product: ป้ายชื่อไม้สลักเลเซอร์
Description: ป้ายชื่อตั้งโต๊ะทำจากไม้สนแท้คุณภาพสูง ผ่านการคัดเลือกเนื้อไม้ที่มีลวดลายสวยงามเป็นธรรมชาติ แข็งแรง ทนทาน และให้ความรู้สึกอบอุ่นเป็นเอกลักษณ์ ผลิตด้วยเทคโนโลยีสลักเลเซอร์ความละเอียดสูง ทำให้ตัวอักษรคมชัด อ่านง่าย และไม่ซีดจางง่าย สามารถสลักชื่อ ตำแหน่ง หรือข้อความตามต้องการ เหมาะสำหรับใช้งานบนโต๊ะทำงาน โต๊ะผู้บริหาร เคาน์เตอร์ต้อนรับ หรือพื้นที่สำนักงานต่าง ๆ
พื้นผิวไม้ผ่านการขัดเรียบและเคลือบป้องกันความชื้น ช่วยยืดอายุการใช้งานและรักษาความสวยงามของเนื้อไม้ในระยะยาว ดีไซน์เรียบหรู สไตล์ทางการ ผสมผสานความทันสมัยและความเป็นธรรมชาติได้อย่างลงตัว เหมาะสำหรับองค์กร บริษัท หรือหน่วยงานที่ต้องการภาพลักษณ์ที่เป็นมืออาชีพ
นอกจากนี้ยังเหมาะสำหรับใช้เป็นของขวัญในโอกาสพิเศษ เช่น ของขวัญเลื่อนตำแหน่ง ของที่ระลึกสำหรับพนักงาน ของขวัญองค์กร หรือของฝากที่มีความหมายเฉพาะบุคคล ช่วยสร้างความประทับใจและสะท้อนความใส่ใจในรายละเอียดได้อย่างดี
Suitable for: ของขวัญเลื่อนตำแหน่ง, ของแจกพนักงาน
Style: ทางการ, อบอุ่น"""
]

# =========================
# 3. LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# ⭐ multilingual แม่นภาษาไทยกว่า all-MiniLM

# =========================
# 4. CREATE EMBEDDINGS
# =========================

print("Creating product embeddings...")

product_embeddings = embed_model.encode(
    records,
    normalize_embeddings=True
)

print("Embedding ready:", len(product_embeddings))


# =========================
# 5. EMBEDDING SEARCH
# =========================

def get_top_candidates(query, top_k=5):

    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(query_vec, product_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    return top_idx.tolist(), scores


# =========================
# 6. LLM RERANK
# =========================

def llm_rerank(query, candidate_idx):

    records_text = "\n\n".join(
        [f"{i}. {records[i]}" for i in candidate_idx]
    )

    prompt = f"""
คุณคือระบบ semantic search

Query: {query}

Products:
{records_text}

เรียง index ที่เกี่ยวข้องที่สุดก่อน
ตอบเป็น JSON list เช่น [2,5,1]
ตอบเฉพาะตัวเลข
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.strip()
        ranking = json.loads(result)

        ranking = [int(i) for i in ranking if i in candidate_idx]

        return ranking

    except Exception as e:
        print("LLM error → fallback:", e)
        return candidate_idx


# =========================
# 7. SEMANTIC SEARCH (Hybrid)
# =========================

def semantic_search(query):

    if not query.strip():
        return None

    candidate_idx, scores = get_top_candidates(query, top_k=5)

    print("Embedding candidates:", candidate_idx)

    reranked = llm_rerank(query, candidate_idx)

    return reranked


# =========================
# 8. PERFORMANCE EVALUATION
# =========================

# ⭐ query → correct answer
test_queries = {
    "ของขวัญเลื่อนตำแหน่ง": [3],
    "ของขวัญผู้ใหญ่": [1],
    "ของแต่งบ้าน": [2],
}

def evaluate_search(k=3):

    print("\n===== PERFORMANCE =====")

    precision_list = []
    recall_list = []
    mrr_list = []
    hit_list = []

    for query, true_idx in test_queries.items():

        result = semantic_search(query)[:k]

        # ---------- Precision ----------
        hit = len(set(result) & set(true_idx))
        precision = hit / k
        recall = hit / len(true_idx)

        precision_list.append(precision)
        recall_list.append(recall)

        # ---------- Hit@K ----------
        hit_list.append(1 if hit > 0 else 0)

        # ---------- MRR ----------
        rr = 0
        for rank, r in enumerate(result):
            if r in true_idx:
                rr = 1 / (rank + 1)
                break

        mrr_list.append(rr)

        print(f"\nQuery: {query}")
        print("Result:", result)
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("MRR:", round(rr, 3))

    print("\n=== AVG ===")
    print("Precision@K:", np.mean(precision_list))
    print("Recall@K:", np.mean(recall_list))
    print("Hit@K:", np.mean(hit_list))
    print("MRR:", np.mean(mrr_list))


# =========================
# 9. RUN PROGRAM
# =========================

if __name__ == "__main__":

    print("===== AI Semantic Search =====")

    while True:
        query = input("\nค้นหา (พิมพ์ exit เพื่อทดสอบ performance): ").strip()

        if query == "":
            break

        if query == "exit":
            evaluate_search()
            continue

        ranking = semantic_search(query)

        if ranking:
            print("\n=== RESULT ===")
            for i in ranking:
                print("\n----------------")
                print(records[i])