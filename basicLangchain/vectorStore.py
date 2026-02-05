from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from uuid import uuid4

env_path = Path(".env/.env")
load_dotenv(dotenv_path=env_path)
chroma_api_key = os.getenv("CHROMA_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


records = [
    "Product: สมุดโน้ตกระดาษคราฟท์ปกแข็งลายวินเทจ\n""Description: สมุดปกแข็ง กระดาษคราฟท์สีน้ำตาลธรรมชาติ 160 แผ่น หนา 100 แกรม พร้อมริบบิ้นคั่นหน้า\n""Suitable for: ของขวัญวันเกิด, ของที่ระลึกงานแต่ง, ของแจกพนักงาน, คนชอบความวินเทจ\n""Style: วินเทจ, เรโทร, อบอุ่น",
    "Product: ปากกาหมึกซึมลายไม้\n""Description: ปากกาหมึกซึมด้ามไม้แท้ เขียนลื่น\n""Suitable for: ของขวัญผู้ใหญ่, งานเกษียณ\n""Style: เรียบหรู",
    "Product: เทียนหอมไขถั่วเหลืองกลิ่นลาเวนเดอร์\nDescription: เทียนหอมธรรมชาติ ไร้เขม่าดำ ช่วยผ่อนคลายและหลับสบาย\nSuitable for: ของขวัญวันเกิด, ของขวัญขึ้นบ้านใหม่, คนชอบแต่งบ้าน\nStyle: มินิมอล, อบอุ่น",
    "Product: โคมไฟตั้งโต๊ะฐานไม้\nDescription: โคมไฟดีไซน์นอร์ดิก ฐานทำจากไม้โอ๊คแท้ ปรับความสว่างได้ 3 ระดับ\nSuitable for: อ่านหนังสือ, ตกแต่งห้องนอน, ของขวัญรับปริญญา\nStyle: นอร์ดิก, มินิมอล",
    "Product: ชุดถ้วยกาแฟเซรามิกทำมือ\nDescription: ถ้วยกาแฟงานปั้นมือ เคลือบสีเอิร์ธโทน เอกลักษณ์ไม่ซ้ำกันในแต่ละใบ\nSuitable for: คนรักกาแฟ, ของที่ระลึกงานแต่ง, ของขวัญผู้ใหญ่\nStyle: คราฟท์, อบอุ่น",
    "Product: กระเป๋าผ้าแคนวาสปักลายใบไม้\nDescription: กระเป๋าผ้าใบหนา ทนทาน ปักลายด้วยมือ ดีไซน์รักษ์โลก\nSuitable for: ของแจกพนักงาน, ของที่ระลึก, ใช้ไปเรียน\nStyle: ธรรมชาติ, เรียบง่าย",
    "Product: นาฬิกาตั้งโต๊ะดิจิทัลลายไม้\nDescription: นาฬิกาบอกเวลาและอุณหภูมิ หน้าจอ LED ซ่อนใต้ผิวไม้\nSuitable for: ตกแต่งออฟฟิศ, ของขวัญปีใหม่\nStyle: โมเดิร์น, มินิมอล",
    "Product: ชุดก้านไม้หอมกระจายกลิ่น\nDescription: Reed Diffuser กลิ่นโอเชี่ยนเฟรช สดชื่นยาวนาน 30 วัน\nSuitable for: ของขวัญงานแต่ง, ของแจกพนักงาน\nStyle: สดชื่น, เรียบหรู",
    "Product: สมุดแพลนเนอร์ปกผ้าลินิน\nDescription: สมุดจดบันทึกรายปี กระดาษถนอมสายตา ปกหุ้มผ้าลินินสีครีม\nSuitable for: คนทำงาน, นักเรียน, ของขวัญวันเกิด\nStyle: มูจิ, มินิมอล",
    "Product: กล่องดนตรีไม้ไขลาน\nDescription: กล่องดนตรีไม้แกะสลัก เพลงคลาสสิก เสียงใสไพเราะ\nSuitable for: ของขวัญวันครบรอบ, ของขวัญเด็ก\nStyle: วินเทจ, คลาสสิก",
    "Product: ผ้าพันคอผ้าไหมพิมพ์ลายไทย\nDescription: ผ้าไหมเนื้อละเอียด พิมพ์ลายไทยประยุกต์ สีสันสดใส\nSuitable for: ของขวัญชาวต่างชาติ, ของขวัญผู้ใหญ่\nStyle: ไทยประยุกต์, หรูหรา",
    "Product: ชุดเครื่องเขียนโลหะสีทอง\nDescription: เซตปากกาและคลิปหนีบกระดาษ สีทองหรูหรา บรรจุในกล่องสวยงาม\nSuitable for: ของขวัญเลื่อนตำแหน่ง, ของขวัญผู้บริหาร\nStyle: ลักชูรี, ทางการ",
    "Product: กระเป๋าสตางค์หนังวัวแท้แบบพับ\nDescription: หนังแท้สัมผัสนุ่ม มีช่องใส่บัตร 8 ช่อง พร้อมช่องซิปใส่เหรียญ\nSuitable for: ของขวัญวันเกิด, ของขวัญผู้ชาย, งานเกษียณ\nStyle: เรียบหรู, คลาสสิก",
    "Product: ชุดชงชาเซรามิกสไตล์ญี่ปุ่น\nDescription: กาน้ำชาพร้อมถ้วย 4 ใบ ลายคลื่นทะเลญี่ปุ่น บรรจุกล่องไม้\nSuitable for: ของขวัญผู้ใหญ่, คนชอบดื่มชา, ของที่ระลึก\nStyle: เซน, ญี่ปุ่น",
    "Product: หมอนอิงกำมะหยี่สีเอิร์ธโทน\nDescription: หมอนอิงนุ่มพิเศษ ปลอกถอดซักได้ ขนาด 45x45 ซม.\nSuitable for: ตกแต่งโซฟา, ของขวัญขึ้นบ้านใหม่\nStyle: โมเดิร์น, อบอุ่น",
    "Product: แผ่นรองเม้าส์หนังสังเคราะห์ขนาดใหญ่\nDescription: แผ่นรองแบบ Desk Mat กันน้ำ ผิวสัมผัสเรียบหรู กว้าง 80 ซม.\nSuitable for: จัดโต๊ะคอม, ของขวัญพนักงานออฟฟิศ\nStyle: มินิมอล, มืออาชีพ",
    "Product: ร่มพับพกพาเคลือบ UV\nDescription: ร่มน้ำหนักเบา กันแดดและกันฝน แข็งแรงทนทานต่อลมแรง\nSuitable for: ของแจกอีเวนต์, ของขวัญพนักงาน\nStyle: ทันสมัย",
    "Product: ชุดปลูกแคคตัส DIY\nDescription: ในชุดประกอบด้วยกระถาง ดิน เมล็ดพันธุ์ และคู่มือการปลูก\nSuitable for: ของขวัญเด็ก, กิจกรรมยามว่าง, คนรักต้นไม้\nStyle: ธรรมชาติ, น่ารัก",
    "Product: ลำโพงไม้บลูทูธพกพา\nDescription: ลำโพงไร้สายดีไซน์ตัวเรือนไม้ ให้เสียงโทนอบอุ่น แบตเตอรี่อึด\nSuitable for: ของขวัญวันเกิด, ตกแต่งโต๊ะทำงาน\nStyle: วินเทจ, ธรรมชาติ",
    "Product: ผ้ากันเปื้อนผ้าลินินสไตล์คาเฟ่\nDescription: ผ้ากันเปื้อนแบบสายไขว้หลัง มีกระเป๋าหน้าใบใหญ่ เนื้อผ้าเกรดเอ\nSuitable for: คนชอบทำอาหาร, เจ้าของร้านกาแฟ, ของขวัญวันแม่\nStyle: มินิมอล, คาเฟ่",
    "Product: หูฟังครอบหูแบบตัดเสียงรบกวน\nDescription: หูฟังไร้สายระบบ ANC เบสแน่น ใส่สบายไม่บีบหู\nSuitable for: คนรักเสียงเพลง, ของขวัญรับปริญญา, คนเดินทางบ่อย\nStyle: เทค, ทันสมัย",
    "Product: ป้ายชื่อไม้สลักเลเซอร์\nDescription: ป้ายชื่อตั้งโต๊ะทำจากไม้สนแท้ สลักชื่อและตำแหน่งด้วยเลเซอร์ความละเอียดสูง\nSuitable for: ของขวัญเลื่อนตำแหน่ง, ของแจกพนักงาน\nStyle: ทางการ, อบอุ่น"
]

metadatas = [
    {"product_id": 1,"category": "stationery","event": "birthday,wedding,corporate","price": 189,"currency": "THB","seller_id": "seller_003","seller_name": "Retro Craft TH","stock": 84},
    {"product_id": 2,"category": "stationery","event": "retirement","price": 890,"currency": "THB","seller_id": "seller_004","seller_name": "WoodCraft","stock": 25},
    {"product_id": 3, "category": "home_decor", "event": "birthday,housewarming", "price": 350, "currency": "THB", "seller_id": "seller_005", "seller_name": "Scent & Soul", "stock": 45},
    {"product_id": 4, "category": "home_decor", "event": "graduation,reading", "price": 790, "currency": "THB", "seller_id": "seller_006", "seller_name": "Light Design", "stock": 20},
    {"product_id": 5, "category": "kitchenware", "event": "wedding,coffee_lover", "price": 420, "currency": "THB", "seller_id": "seller_007", "seller_name": "Ceramic Studio", "stock": 15},
    {"product_id": 6, "category": "fashion", "event": "corporate,souvenir", "price": 250, "currency": "THB", "seller_id": "seller_008", "seller_name": "Green Bag TH", "stock": 100},
    {"product_id": 7, "category": "gadget", "event": "new_year,office", "price": 550, "currency": "THB", "seller_id": "seller_009", "seller_name": "Woody Tech", "stock": 30},
    {"product_id": 8, "category": "lifestyle", "event": "wedding,corporate", "price": 390, "currency": "THB", "seller_id": "seller_010", "seller_name": "Aroma Fresh", "stock": 60},
    {"product_id": 9, "category": "stationery", "event": "birthday,student", "price": 290, "currency": "THB", "seller_id": "seller_011", "seller_name": "Minimal Note", "stock": 85},
    {"product_id": 10, "category": "gift", "event": "anniversary,children", "price": 1200, "currency": "THB", "seller_id": "seller_012", "seller_name": "Music Box Shop", "stock": 12},
    {"product_id": 11, "category": "fashion", "event": "foreigner,elderly", "price": 1500, "currency": "THB", "seller_id": "seller_013", "seller_name": "Thai Silk Co.", "stock": 25},
    {"product_id": 12, "category": "stationery", "event": "promotion,executive", "price": 2100, "currency": "THB", "seller_id": "seller_014", "seller_name": "Elite Office", "stock": 8},
    {"product_id": 13, "category": "fashion", "event": "birthday,retirement", "price": 1250, "currency": "THB", "seller_id": "seller_015", "seller_name": "Leather Master", "stock": 40},
    {"product_id": 14, "category": "kitchenware", "event": "elderly,souvenir", "price": 980, "currency": "THB", "seller_id": "seller_016", "seller_name": "Zen Ceramic", "stock": 18},
    {"product_id": 15, "category": "home_decor", "event": "housewarming", "price": 350, "currency": "THB", "seller_id": "seller_017", "seller_name": "Soft Living", "stock": 55},
    {"product_id": 16, "category": "office_supplies", "event": "corporate,office_setup", "price": 490, "currency": "THB", "seller_id": "seller_018", "seller_name": "Desk Decor", "stock": 120},
    {"product_id": 17, "category": "lifestyle", "event": "corporate,giveaway", "price": 290, "currency": "THB", "seller_id": "seller_019", "seller_name": "Everyday Carry", "stock": 200},
    {"product_id": 18, "category": "gardening", "event": "hobby,children", "price": 199, "currency": "THB", "seller_id": "seller_020", "seller_name": "Green Thumb", "stock": 65},
    {"product_id": 19, "category": "electronics", "event": "birthday,office", "price": 1590, "currency": "THB", "seller_id": "seller_021", "seller_name": "Wooden Sound", "stock": 22},
    {"product_id": 20, "category": "fashion", "event": "cooking,mother_day", "price": 450, "currency": "THB", "seller_id": "seller_022", "seller_name": "Cafe Wear", "stock": 38},
    {"product_id": 21, "category": "electronics", "event": "graduation,travel", "price": 3200, "currency": "THB", "seller_id": "seller_023", "seller_name": "Audio Tech", "stock": 15},
    {"product_id": 22, "category": "office_supplies", "event": "promotion,corporate", "price": 590, "currency": "THB", "seller_id": "seller_024", "seller_name": "Craft Sign", "stock": 50}
]

ollama_embeddings_model = OllamaEmbeddings(
    model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
)

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 1536 dim
)

google_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key="AIzaSyAVus7xiFN6rZWXz4uKoqT2NvVe1rnr6v8"
)

from langchain_chroma import Chroma
import chromadb
import uuid

client = chromadb.CloudClient(
  api_key=chroma_api_key,
  tenant='f69a80c1-ef1c-467e-be0a-4295c55b5ffb',
  database='testing'
)

vectorstore = Chroma(
    collection_name="ollama_collection",
    embedding_function=ollama_embeddings_model,
    client=client
)
# เพิ่มข้อมูลลง Chroma
# vectorstore.add_texts(
#     texts=records,
#     metadatas=metadatas,
#     ids=[str(uuid4()) for _ in records]
# )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# document_1 = Document(
#     page_content="""
# Product: สมุดโน้ตกระดาษคราฟท์ปกแข็งลายวินเทจ
# Description: สมุดปกแข็ง กระดาษคราฟท์สีน้ำตาลธรรมชาติ 160 แผ่น หนา 100 แกรม พร้อมริบบิ้นคั่นหน้า
# Suitable for: ของขวัญวันเกิด, ของที่ระลึกงานแต่ง, ของแจกพนักงาน, คนชอบความวินเทจ
# Style: วินเทจ, เรโทร, อบอุ่น
# """,
#     metadata={
#         "product_id": 1, # id ที่ต่อกับ user
#         "category": "stationery",
#         "event": "birthday,wedding,corporate",
#         "price": 189,
#         "currency": "THB",
#         "seller_id": "seller_003",
#         "seller_name": "Retro Craft TH",
#         "stock": 84
#     }
# )

# document_2 = Document(
#     page_content="""
# Product: แก้วเซรามิค 11 oz หม้อไฟลายการ์ตูน
# Description: แก้วเซรามิคเคลือบด้าน ภาพพิมพ์คุณภาพสูง ทนความร้อน ล้างเครื่องล้างจานได้
# Suitable for: ของขวัญให้เพื่อน, ของขวัญคู่รัก, ของแจกงานอีเวนต์, คนชอบอนิเมะ
# Style: น่ารัก, คาวาอี้, สีสันสดใส
# """,
#     metadata={
#         "product_id": 2,
#         "category": "mug",
#         "event": "friendship,valentine,event",
#         "price": 129,
#         "currency": "THB",
#         "seller_id": "seller_007",
#         "seller_name": "CutePrint Studio",
#         "stock": 245
#     }
# )

# document_3 = Document(
#     page_content="""
# Product: ชุดปากกาเจลญี่ปุ่น 5 สี + กล่องของขวัญ
# Description: ปากกาเจล Pilot Juice 0.38mm 5 สีพาสเทล ในกล่องของขวัญพร้อมโบว์
# Suitable for: ของขวัญนักเรียน, ของขวัญครู, Secret Santa, ของแจกงานจบการศึกษา
# Style: น่ารัก, พาสเทล, ของขวัญชิ้นเล็ก
# """,
#     metadata={
#         "product_id": 3,
#         "category": "stationery",
#         "event": "graduation,teacher,student",
#         "price": 259,
#         "currency": "THB",
#         "seller_id": "seller_002",
#         "seller_name": "Stationery Japan",
#         "stock": 67
#     }
# )

# document_4 = Document(
#     page_content="""
# Product: สมุดแพลนเนอร์รายเดือน A5 ปี 2026 ลายดอกไม้แห้ง
# Description: แพลนเนอร์ปี 2026 กระดาษถนอมสายตา ผ้าครึ่งแข็ง หน้าปกฟอยล์ร้อน
# Suitable for: คนรักการแพลนชีวิต, ของขวัญปีใหม่, ของขวัญให้ตัวเอง
# Style: อ่อนโยน, เรียบหรู, ธรรมชาติ
# """,
#     metadata={
#         "product_id": 4,
#         "category": "planner",
#         "event": "newyear,selfgift,corporate",
#         "price": 390,
#         "currency": "THB",
#         "seller_id": "seller_015",
#         "seller_name": "Paper & Bloom",
#         "stock": 38
#     }
# )

# document_5 = Document(
#     page_content="""
# Product: ตุ๊กตาหมีตัวเล็ก 15 ซม. ถือป้ายขอแต่งงาน
# Description: ตุ๊กตาหมีนุ่ม ใส่เสื้อผ้าสีขาว ถือป้าย "จะแต่งงานกับพี่ได้มั้ย"
# Suitable for: ของขวัญขอแต่งงาน, Valentine, Anniversary, Surprise proposal
# Style: โรแมนติก, น่ารัก, เสน่ห์
# """,
#     metadata={
#         "product_id": 5,
#         "category": "gift",
#         "event": "proposal,valentine,anniversary",
#         "price": 290,
#         "currency": "THB",
#         "seller_id": "seller_009",
#         "seller_name": "Bear & Love",
#         "stock": 112
#     }
# )

# document_6 = Document(
#     page_content="""
# Product: ขวดน้ำสแตนเลส 500ml คู่รักลายมินิมอล
# Description: ขวดน้ำเก็บความเย็น 24 ชม. + ความร้อน 12 ชม. มี 2 สี สลักชื่อได้
# Suitable for: ของขวัญคู่รัก, ของขวัญเพื่อนซี้, ของขวัญวันครบรอบ
# Style: มินิมอล, คู่รัก, ทันสมัย
# """,
#     metadata={
#         "product_id": 6,
#         "category": "tumbler",
#         "event": "couple,anniversary,friends",
#         "price": 480,
#         "currency": "THB",
#         "seller_id": "seller_004",
#         "seller_name": "Minimal Drinkware",
#         "stock": 95
#     }
# )

# document_7 = Document(
#     page_content="""
# Product: สมุดสเก็ตช์ลายน้ำอ่อน 200 แผ่น กระดาษคราฟท์ขาว
# Description: สมุดสเก็ตช์กระดาษ 120 แกรม ไม่มีเส้นลาย ใส่กระเป๋าได้พกง่าย
# Suitable for: ศิลปิน, นักวาดรูป, ของขวัญให้คนชอบวาดรูป
# Style: ศิลปะ, มินิมอล, คราฟท์
# """,
#     metadata={
#         "product_id": 7,
#         "category": "sketchbook",
#         "event": "art,graduation,birthday",
#         "price": 220,
#         "currency": "THB",
#         "seller_id": "seller_011",
#         "seller_name": "Artisan Paper",
#         "stock": 156
#     }
# )

# document_8 = Document(
#     page_content="""
# Product: ชุดสติ๊กเกอร์กันน้ำ 50 ชิ้น ลายแมวเหมียวหลากอารมณ์
# Description: สติ๊กเกอร์กันน้ำคุณภาพสูง ตัดขอบเรียบ ใช้ติดโน้ตบุ๊ค แล็ปท็อป ขวดน้ำ
# Suitable for: ของแจกงานอีเวนต์, ของแถม, คนชอบสติ๊กเกอร์, วัยรุ่น
# Style: น่ารัก, แมว, สีสันสดใส
# """,
#     metadata={
#         "product_id": 8,
#         "category": "sticker",
#         "event": "event,general,teen",
#         "price": 99,
#         "currency": "THB",
#         "seller_id": "seller_008",
#         "seller_name": "Meow Sticker Club",
#         "stock": 780
#     }
# )

# document_9 = Document(
#     page_content="""
# Product: พวงกุญแจหนังแท้ สลักชื่อ + วันสำคัญ
# Description: พวงกุญแจหนังแท้เกรด A สลักชื่อ/วันที่ด้วยเลเซอร์ มีกล่องของขวัญ
# Suitable for: ของขวัญวันเกิด, ของที่ระลึกจบการศึกษา, ของขวัญพนักงาน
# Style: เรียบหรู, มินิมอล, งานฝีมือ
# """,
#     metadata={
#         "product_id": 9,
#         "category": "keychain",
#         "event": "birthday,graduation,corporate",
#         "price": 350,
#         "currency": "THB",
#         "seller_id": "seller_005",
#         "seller_name": "Leather Craft TH",
#         "stock": 63
#     }
# )

# document_10 = Document(
#     page_content="""
# Product: Limited Edition Notebook
# Description: Notebook with inspirational quotes on every page.
# Suitable for: Appreciation gifts, graduation souvenirs, company giveaways.
# Style: Minimal, elegant, lightweight.
# """,
#     metadata={
#         "product_id": 10,
#         "category": "stationery",
#         "event": "general,graduation",
#         "price": 149,
#         "currency": "THB",
#         "seller_id": "seller_001",
#         "seller_name": "ABC Stationery",
#         "stock": 120
#     }
# )

# documents = [
#     document_1,
#     document_2,
#     document_3,
#     document_4,
#     document_5,
#     document_6,
#     document_7,
#     document_8,
#     document_9,
#     document_10,
# ]

# generate uid สำหรับ vector database อันนี้ช่างแม่งเราไม่ได้ยุ่ง Database มันเชื่อมกันเองออโต้
# uuids = [str(uuid4()) for _ in range(len(documents))]
# ----------------------------------------

# results = collection.query(
#     query_texts=[
#         "สินค้าเกี่ยวกับอะไร"
#     ],
#     n_results= 5
# )

# for i, query_results in enumerate(results["documents"]):
#     print(f"\nQuery {i}")
#     print("\n".join(query_results))

# collection.add(ids=uuid4, documents=documents)

# vector_store = Chroma(
#     collection_name="testCollection",
#     embedding_function=embeddings_model,
#     persist_directory="./basicLangchain/testing_chroma_db",
#     collection_metadata={"hnsw:space": "cosine"}
# )

# เซฟลง Database --------------
# vector_store.add(documents=documents, ids=uuids)
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