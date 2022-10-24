from pdfid import *
import json
import os
import collections
import pandas as ps

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

def dir_analysis(dirname):
	file_name = "result_" + os.path.basename(dirname[:len(dirname)-1]) + ".txt"
	file = open(file_name, "w")
	counter = collections.Counter()
	i = 0

	print("Analysis is started...")
	
	for filename in os.listdir(dirname):
		f = os.path.join(dirname, filename)
		if os.path.isfile(f):
			dic = pdf_analysis(f)
			counter.update(dic)
			print("..." + str(i))
			i+=1
	res = dict(counter)
	print("...analysis is finished")

	print(res)
	file.write(json.dumps(res))
	file.close()
	return file_name

if __name__ == '__main__':

	directories = ["../../pdf_data/malware_pdf_cve_sorted_173_files/", "../../pdf_data/clean_pdf_9000_files/"]#, "../../pdf_data/malware_pdf_pre_04-2011_10982_files/"]

	for dire in directories:
		out = dir_analysis(dire)
		print("Produced " + out)
