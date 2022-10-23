from pdfid import *
import json
import os
import collections

def pdf_analysis(filename):
	xml = PDFiD(filename)
	doc = PDFiD2JSON(xml, False)

	y = json.loads(doc)
	js = json.loads(json.dumps(y[0]))

	dic = {}
	for w in js["pdfid"]["keywords"]["keyword"]:
		name = w["name"]
		count = w["count"]
		dic.update({name:count})
	
	return dic

if __name__ == '__main__':

	#directory = "../../../pro/"
	directory = "../../pdf_data/clean_pdf_9000_files/"
	counter = collections.Counter()

	print("Analysis is started...")
	
	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		if os.path.isfile(f):
			dic = pdf_analysis(f)
			counter.update(dic)
			print(".")
	res = dict(counter)
	print("...analysis is finished")
	print(res)
