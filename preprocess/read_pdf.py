from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for _,page in enumerate(reader.pages):
        text += page.extract_text()
    return text