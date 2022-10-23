from PDFConsole import PDFConsole
from PDFCore import PDFParser

print("start")
pdfParser = PDFParser()
ret, pdf = pdfParser.parse("cv.pdf", True, False)
print(pdf.getStats())
print("Constains JS: " + pdf.containsJS())