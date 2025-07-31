# for OCR
import pathlib
from pathlib import Path

from pdf2image import convert_from_path

# to iterate over multiple PDF files
from glob import glob

# for creating a text selectable pdf
import pytesseract
from pypdf import PdfWriter


# convert pdf to png
def pdf_to_image(pdf_path, output_folder="rag/pdf_images"):
    """
    function to convert pdfs to images
    """
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get base name (e.g., "mentoring_report" from "mentoring_report.pdf")
    base_name = pdf_path.stem
    expected_first_image = output_folder / f"{base_name}_page_1.png"

    if expected_first_image.exists():
        print(f"✅ Skipping {pdf_path.name} (already converted)")
        return 0

    # Convert PDF to list of images (one per page)
    images = convert_from_path(str(pdf_path))

    for i, img in enumerate(images):
        output_file = output_folder / f"{base_name}_page_{i+1}.png"
        img.save(output_file, "PNG")

    print(f"✅ Converted {pdf_path.name} ({len(images)} pages)")
    return len(images)

# convert all existing items in the folder
ALL_PDF_PATHS = glob("rag/raw_pdfs/*.pdf")

from pdf2image.exceptions import PDFPageCountError

for pdf_file in ALL_PDF_PATHS:

    try:
        print(f"Processing: {pdf_file}")
        pdf_to_image(pdf_path=pdf_file, output_folder="rag/pdf_images")
    except PDFPageCountError:
        print(f"❌ Skipping unreadable PDF: {pdf_file}")
    except Exception as e:
        print(f"❌ Unexpected error with {pdf_file}: {e}")

print(f"Successfully converted {len(ALL_PDF_PATHS)} pdf files into png images :)")


# conversion into text selectable pdfs
def ocr_images_to_searchable_pdf(stem=None, image_folder="rag/pdf_images", output_folder="rag/processed_pdfs"):
    
    image_folder=Path(image_folder)
    output_folder=Path(output_folder)

    page_images = sorted(
        image_folder.glob(f"{stem}_page_*.png"),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    if not page_images:
            print(f"No images found for stem: {stem}")
            return

    merger = PdfWriter()

    for img_path in page_images:
        raw_pdf = pytesseract.image_to_pdf_or_hocr(str(img_path), extension="pdf")

        temp_pdf_path = img_path.parent / f"__temp_{img_path.stem}.pdf"

        with open(temp_pdf_path, "wb") as f:
            f.write(raw_pdf)

        merger.append(str(temp_pdf_path))
        temp_pdf_path.unlink()  # deletes temp file to keep script clean
    
    output_pdf =output_folder /f"{stem}_ocr.pdf"
    merger.write(str(output_pdf))
    merger.close()

image_folder = Path("rag/pdf_images")
output_folder = Path("rag/processed_pdfs")
output_folder.mkdir(parents=True, exist_ok=True)

# Find all _page_1.png files to get one entry per document
unique_page_ones = image_folder.glob("*_page_1.png")
stems = [p.stem.rsplit("_page_", 1)[0] for p in unique_page_ones]

for stem in stems:
    try:
        print(f"🔄 Processing: {stem}")
        ocr_images_to_searchable_pdf(stem, image_folder=image_folder, output_folder=output_folder)
    except Exception as e:
        print(f"❌ Error with {stem}: {e}")
print(f"Successfully created {len(stems)} searchable text pdfs :)")


