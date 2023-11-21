import PyPDF2

# Abre el archivo PDF
pdf_file = open("invoice.pdf", "rb")

# Crea un lector de PDF
reader = PyPDF2.PdfReader(pdf_file)

# Obtiene el texto del primer página
page_number=0
text = reader.pages[page_number].extract_text()

# Imprime el texto
print(text)
