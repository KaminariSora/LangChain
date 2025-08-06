from langchain_community.document_loaders import PyPDFLoader

file_path = './Data/report_grade.pdf'
loader = PyPDFLoader(file_path)
result = loader.load()

print(result)


# [Document
#     (metadata=
#         {
#             'producer': 'Thai PDF 1.00T', 'creator': 'PyPDF', 'creationdate': 'D:20240131163846', 'source': './Data/report_grade.pdf', 'total_pages': 1, 'page': 0, 'page_label': '1'
#         },
#         page_content='มหาวิทยาลัยศรีนครินทรวิโรฒ ทบ. 12\n114 สุขุมวิท 23 เขตวัฒนา กรุงเทพ ฯ 10110\nใบแจ้งผลการศึกษา ประจ           จำภาคเรียนที่ 1 ปีการศึกษา 2566\n65109010208 นายนนทชา หวลจิตต์ (EG28)\nอาจารย์ที่ปรึกษา : ผู้ช่วยศาสตราจารย์วัชรชัย วิริยะสุทธิวงศ์\nจำนวนหน่วยกิตสะสมที่                         สอบได้เเล้ว66 หน่วยกิตระดับขั้นเฉลี่ยสะสมภาคเรียนที่ผ่านมา3.20\nที่ วิชา หน่วยกิตเกรดหมายเหตุ\nลงทะเบียนได้สอบได้ นำมาคำนวณคะแนน ระดับขั้นเฉลี่ย\nผลการเ                          เรียนในภาคเรียนนี้23 หน่วยกิต23 หน่วยกิต23 หน่วยกิต80.50 3.50\nผลการเรียนสะสมถึงภาคปัจจุบัน66 หน่วยกิต66 หน่วยกิต66 หน่วยกิต211.50 3.20\nหมายเหตุ 1. ในรา                     ายการวิชาใดที่ยังไม่มีเกรดให้ติดต่อาจารย์ผู้สอน\n1 CPE200 English for Engineers 1 B01 2 B+\n2 CPE202 Linear Algebra for Computer Engineering B01 3 A\n3 C           CPE230 Database System Design and Management B01 3 B\n4 CPE240 Electronics for Computer Engineering B01 3 B+\n5 CPE241 Computer  Architecture and Organization B01 3 C+\n6 CPE410 Artificial Neural Networks B01 3 B+\n7 SWU195 Creative Citizen for Society B23 3 A\n8 SWU196 Science and Art of Sustainable Social Development B23 3 A'
#     )
# ]