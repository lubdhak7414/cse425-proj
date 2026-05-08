from pathlib import Path
from PyPDF2 import PdfReader

base = Path("requirements")
pdfs = sorted(base.glob("*.pdf"))
if not pdfs:
    print("No PDF files found in requirements/")
    raise SystemExit(1)

for pdf in pdfs:
    reader = PdfReader(str(pdf))
    out = base / (pdf.stem + ".txt")
    with out.open("w", encoding="utf-8") as f:
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                f.write(text)
                f.write("\n\n")
    print(f"Wrote {out}")
