import pypdf

pdf_path = r"C:\Users\vardh\Desktop\RAS\Path_Planning_Algorithm_for_an_Autonomous_Electric_Wheelchair_in_Hospitals.pdf"
try:
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
            
    with open('extracted_paper.txt', 'w', encoding='utf-8') as out:
        out.write(text)
    print("Extracted successfully.")
except Exception as e:
    print(f"Error: {e}")
