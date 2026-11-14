
import fitz
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

from PIL import Image
from io import BytesIO


pdf_path = "data/fluid_mechanics_fixed.pdf"

doc = fitz.open(pdf_path)

for page_number, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    image = Image.open(
        BytesIO(pix.tobytes("png"))
    )

    text = pytesseract.image_to_string(image)

    print(f"\n--- PAGE {page_number + 1} ---")
    print(text[:1000])

doc.close()
