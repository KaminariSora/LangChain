from langchain_ollama import OllamaEmbeddings
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="Handmade ceramic mug with minimalist design, suitable for wedding souvenirs.",
    metadata={"category": "souvenir", "event": "wedding", "price": 250},
    id=1,
)

document_2 = Document(
    page_content="Customized acrylic keychain that can be printed with names or logos.",
    metadata={"category": "keychain", "event": "general", "price": 89},
    id=2,
)

document_3 = Document(
    page_content="Premium fabric tote bag with custom message, ideal for corporate gifts.",
    metadata={"category": "bag", "event": "corporate", "price": 199},
    id=3,
)

document_4 = Document(
    page_content="Scented candle set with elegant packaging, commonly used for funeral ceremonies.",
    metadata={"category": "candle", "event": "funeral", "price": 180},
    id=4,
)

document_5 = Document(
    page_content="Graduation teddy bear with customizable sash and year.",
    metadata={"category": "doll", "event": "graduation", "price": 459},
    id=5,
)

document_6 = Document(
    page_content="Wooden USB flash drive engraved with company logo for business souvenirs.",
    metadata={"category": "usb", "event": "corporate", "price": 320},
    id=6,
)

document_7 = Document(
    page_content="Thank-you gift box containing tea and snacks, popular for housewarming events.",
    metadata={"category": "gift_box", "event": "housewarming", "price": 399},
    id=7,
)

document_8 = Document(
    page_content="Anime-style illustration postcard set, designed by independent artists.",
    metadata={"category": "art", "event": "anime", "price": 120},
    id=8,
)

document_9 = Document(
    page_content="VTuber-themed sticker pack with holographic finish.",
    metadata={"category": "sticker", "event": "vtuber", "price": 79},
    id=9,
)

document_10 = Document(
    page_content="Limited edition notebook with inspirational quotes, suitable as appreciation gifts.",
    metadata={"category": "stationery", "event": "general", "price": 149},
    id=10,
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
# generate uid สำหรับใส่ vector database
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

# results = vector_store.similarity_search(
#     "อยากได้สินค้าในทีม Vtuber",
#     k=2,
#     # filter={"source": "tweet"},
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

results = vector_store.similarity_search_with_score(
    "กระเป๋า",
    k=2
)

for doc, score in results:
    # chroma ใช้ cosin distance แปลว่า score ยิ่งน้อยยิ่งดี
    print(score, doc.page_content)
