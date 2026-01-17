from langchain_ollama import OllamaEmbeddings
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="""
Product: สมุดโน้ตกระดาษคราฟท์ปกแข็งลายวินเทจ
Description: สมุดปกแข็ง กระดาษคราฟท์สีน้ำตาลธรรมชาติ 160 แผ่น หนา 100 แกรม พร้อมริบบิ้นคั่นหน้า
Suitable for: ของขวัญวันเกิด, ของที่ระลึกงานแต่ง, ของแจกพนักงาน, คนชอบความวินเทจ
Style: วินเทจ, เรโทร, อบอุ่น
""",
    metadata={
        "product_id": 1, # id ที่ต่อกับ user
        "category": "stationery",
        "event": "birthday,wedding,corporate",
        "price": 189,
        "currency": "THB",
        "seller_id": "seller_003",
        "seller_name": "Retro Craft TH",
        "stock": 84
    }
)

document_2 = Document(
    page_content="""
Product: แก้วเซรามิค 11 oz หม้อไฟลายการ์ตูน
Description: แก้วเซรามิคเคลือบด้าน ภาพพิมพ์คุณภาพสูง ทนความร้อน ล้างเครื่องล้างจานได้
Suitable for: ของขวัญให้เพื่อน, ของขวัญคู่รัก, ของแจกงานอีเวนต์, คนชอบอนิเมะ
Style: น่ารัก, คาวาอี้, สีสันสดใส
""",
    metadata={
        "product_id": 2,
        "category": "mug",
        "event": "friendship,valentine,event",
        "price": 129,
        "currency": "THB",
        "seller_id": "seller_007",
        "seller_name": "CutePrint Studio",
        "stock": 245
    }
)

document_3 = Document(
    page_content="""
Product: ชุดปากกาเจลญี่ปุ่น 5 สี + กล่องของขวัญ
Description: ปากกาเจล Pilot Juice 0.38mm 5 สีพาสเทล ในกล่องของขวัญพร้อมโบว์
Suitable for: ของขวัญนักเรียน, ของขวัญครู, Secret Santa, ของแจกงานจบการศึกษา
Style: น่ารัก, พาสเทล, ของขวัญชิ้นเล็ก
""",
    metadata={
        "product_id": 3,
        "category": "stationery",
        "event": "graduation,teacher,student",
        "price": 259,
        "currency": "THB",
        "seller_id": "seller_002",
        "seller_name": "Stationery Japan",
        "stock": 67
    }
)

document_4 = Document(
    page_content="""
Product: สมุดแพลนเนอร์รายเดือน A5 ปี 2026 ลายดอกไม้แห้ง
Description: แพลนเนอร์ปี 2026 กระดาษถนอมสายตา ผ้าครึ่งแข็ง หน้าปกฟอยล์ร้อน
Suitable for: คนรักการแพลนชีวิต, ของขวัญปีใหม่, ของขวัญให้ตัวเอง
Style: อ่อนโยน, เรียบหรู, ธรรมชาติ
""",
    metadata={
        "product_id": 4,
        "category": "planner",
        "event": "newyear,selfgift,corporate",
        "price": 390,
        "currency": "THB",
        "seller_id": "seller_015",
        "seller_name": "Paper & Bloom",
        "stock": 38
    }
)

document_5 = Document(
    page_content="""
Product: ตุ๊กตาหมีตัวเล็ก 15 ซม. ถือป้ายขอแต่งงาน
Description: ตุ๊กตาหมีนุ่ม ใส่เสื้อผ้าสีขาว ถือป้าย "จะแต่งงานกับพี่ได้มั้ย"
Suitable for: ของขวัญขอแต่งงาน, Valentine, Anniversary, Surprise proposal
Style: โรแมนติก, น่ารัก, เสน่ห์
""",
    metadata={
        "product_id": 5,
        "category": "gift",
        "event": "proposal,valentine,anniversary",
        "price": 290,
        "currency": "THB",
        "seller_id": "seller_009",
        "seller_name": "Bear & Love",
        "stock": 112
    }
)

document_6 = Document(
    page_content="""
Product: ขวดน้ำสแตนเลส 500ml คู่รักลายมินิมอล
Description: ขวดน้ำเก็บความเย็น 24 ชม. + ความร้อน 12 ชม. มี 2 สี สลักชื่อได้
Suitable for: ของขวัญคู่รัก, ของขวัญเพื่อนซี้, ของขวัญวันครบรอบ
Style: มินิมอล, คู่รัก, ทันสมัย
""",
    metadata={
        "product_id": 6,
        "category": "tumbler",
        "event": "couple,anniversary,friends",
        "price": 480,
        "currency": "THB",
        "seller_id": "seller_004",
        "seller_name": "Minimal Drinkware",
        "stock": 95
    }
)

document_7 = Document(
    page_content="""
Product: สมุดสเก็ตช์ลายน้ำอ่อน 200 แผ่น กระดาษคราฟท์ขาว
Description: สมุดสเก็ตช์กระดาษ 120 แกรม ไม่มีเส้นลาย ใส่กระเป๋าได้พกง่าย
Suitable for: ศิลปิน, นักวาดรูป, ของขวัญให้คนชอบวาดรูป
Style: ศิลปะ, มินิมอล, คราฟท์
""",
    metadata={
        "product_id": 7,
        "category": "sketchbook",
        "event": "art,graduation,birthday",
        "price": 220,
        "currency": "THB",
        "seller_id": "seller_011",
        "seller_name": "Artisan Paper",
        "stock": 156
    }
)

document_8 = Document(
    page_content="""
Product: ชุดสติ๊กเกอร์กันน้ำ 50 ชิ้น ลายแมวเหมียวหลากอารมณ์
Description: สติ๊กเกอร์กันน้ำคุณภาพสูง ตัดขอบเรียบ ใช้ติดโน้ตบุ๊ค แล็ปท็อป ขวดน้ำ
Suitable for: ของแจกงานอีเวนต์, ของแถม, คนชอบสติ๊กเกอร์, วัยรุ่น
Style: น่ารัก, แมว, สีสันสดใส
""",
    metadata={
        "product_id": 8,
        "category": "sticker",
        "event": "event,general,teen",
        "price": 99,
        "currency": "THB",
        "seller_id": "seller_008",
        "seller_name": "Meow Sticker Club",
        "stock": 780
    }
)

document_9 = Document(
    page_content="""
Product: พวงกุญแจหนังแท้ สลักชื่อ + วันสำคัญ
Description: พวงกุญแจหนังแท้เกรด A สลักชื่อ/วันที่ด้วยเลเซอร์ มีกล่องของขวัญ
Suitable for: ของขวัญวันเกิด, ของที่ระลึกจบการศึกษา, ของขวัญพนักงาน
Style: เรียบหรู, มินิมอล, งานฝีมือ
""",
    metadata={
        "product_id": 9,
        "category": "keychain",
        "event": "birthday,graduation,corporate",
        "price": 350,
        "currency": "THB",
        "seller_id": "seller_005",
        "seller_name": "Leather Craft TH",
        "stock": 63
    }
)

document_10 = Document(
    page_content="""
Product: Limited Edition Notebook
Description: Notebook with inspirational quotes on every page.
Suitable for: Appreciation gifts, graduation souvenirs, company giveaways.
Style: Minimal, elegant, lightweight.
""",
    metadata={
        "product_id": 10,
        "category": "stationery",
        "event": "general,graduation",
        "price": 149,
        "currency": "THB",
        "seller_id": "seller_001",
        "seller_name": "ABC Stationery",
        "stock": 120
    }
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
# generate uid สำหรับ vector database อันนี้ช่างแม่งเราไม่ได้ยุ่ง Database มันเชื่อมกันเองออโต้
uuids = [str(uuid4()) for _ in range(len(documents))]
# ----------------------------------------

embeddings_model = OllamaEmbeddings(
    model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
)

vector_store = Chroma(
    collection_name="testCollection",
    embedding_function=embeddings_model,
    persist_directory="./basicLangchain/testing_chroma_db"
)
# เซฟลง Database --------------
# vector_store.add_documents(documents=documents, ids=uuids)
# -----------------------------

# test similarity search -----------
# results = vector_store.similarity_search_with_score(
#     "Product สติ๊กเกอร์",
#     k=2,
#     filter={'category': 'sticker'}
# )

# for doc, score in results:
#     # chroma ใช้ cosin distance แปลว่า score ยิ่งน้อยยิ่งดี
#     print(f"score: ", score, doc.page_content)
#     print("-----------------")
# ----------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)