from PyPDF2 import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Example usage  
if __name__ == "__main__":
    text = extract_text("data/policy.pdf")
    print(text[:500])  # Show first 500 chars