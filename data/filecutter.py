from pypdf import PdfReader, PdfWriter

def extract_first_n_pages(input_pdf, output_pdf, n_pages=6):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    pages_to_extract = min(n_pages, total_pages)

    for i in range(pages_to_extract):
        writer.add_page(reader.pages[i])

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"✅ Created '{output_pdf}' with first {pages_to_extract} pages.")

if __name__ == "__main__":
    input_pdf_path = "Rev 2 oxford-partial-knee-microplasty-instrumentation-surgical-technique1 (1).pdf"      # change this
    output_pdf_path = "output2.pdf"    # change this

    extract_first_n_pages(
        input_pdf=input_pdf_path,
        output_pdf=output_pdf_path,
        n_pages=6
    )
