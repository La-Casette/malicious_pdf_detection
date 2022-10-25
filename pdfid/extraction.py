from pdfid import *
import json
import os

def pdf_analysis(filename):
    xml = PDFiD(filename)
    doc = PDFiD2JSON(xml, False)

    y = json.loads(doc)
    js = json.loads(json.dumps(y[0]))

    l = []
    for w in js["pdfid"]["keywords"]["keyword"]:
        l.append(w["count"])
        
    return l

def dir_analysis(dirname):
    mat = []

    for filename in os.listdir(dirname):
        f = os.path.join(dirname, filename)
        if os.path.isfile(f):
            l = pdf_analysis(f)
            mat.append(l)
            #print("..." + str(i))
            #i += 1
            
    return mat