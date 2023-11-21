import tabula

# Abre el archivo PDF
pdf_file = "table.pdf"

# Extrae las tablas del PDF
tables = tabula.read_pdf(pdf_file, pages="all")
print(tables)
# Imprime las tablas
for table in tables:
    print(table)
