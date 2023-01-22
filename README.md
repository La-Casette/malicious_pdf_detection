# Machine Learning techniques for malicious PDF detection

This project presents an exhaustive analysis of machine learning techniques for detecting malicious PDF files. By comparing the performance of four binary classification models - Support Vector Machines (SVM), Decision Trees, Random Forests and K-Nearest Neighbors (KNN) - we demonstrate the effectiveness of these approaches in achieving high Accuracy and low False Negative Rates.

What makes this project stand out is its focus on the machine learning models themselves, including the careful selection of hyperparameters to optimize their performance. Additionally, the inclusion of KNN as a valid model is rare in the field, and the analysis of evasion techniques adds another layer of depth to the study. Overall this paper, realized for the MALIS course at EURECOM (year 2022/2023), offers a comprehensive look at the use of machine learning for malicious PDF detection and the potential vulnerabilities of these models.

## Project Content

- `data/`: Output of the data extraction pipeline in `numpy` and `pandas` format, notebook with the functions used to split in Train and Test the dataset.
- `models/`: Jupyter notebooks where all the models were implemented and tested.
- `pdf_data_enc/`: Encrypted PDF files of the Contagio dataset.
- `pdfid/`: Library functions of the PDFiD tool
- `utils/`: Model Evaluation and Data Import functions.
- `report/`: LaTeX report
- `poster/`: Presentation slide

## Poster

![Poster](/poster/poster.png)