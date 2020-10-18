## Auto-grading
An auto-grading system for handwritten digits implemented by Python.  
Authors: Jephian Lin (maintainer) and Chan-Yu Pan (main).  

Necessary Requirements : tensorflow 2.1.0, pdf2image 1.13.1, Pillow 7.0.0 and pyzbar 0.1.8.  

## Repo contents
* __main.ipynb__ : Main file for excuting this auto-grading system.  
* __extractor.py__ : This script includes the functions for extracting the scanned file.  
* __ocr.py__ : This script holds all the code to create OCR model.  
* __annotator.py__ : This script includes the functions for grading on their examination paper.
* __OCR_mdl.h5__ : This file is the trained model for OCR system.  
* __font_style__ : This folder includes the necessary files about font style for annotator to grade.  
* __nsysu-digits__ : This folder is our handwritten digit database. All images are grayscale and the size is 28*28.  

## How to Use
If you have installed jupyter notebook, run the file `main.ipynb` on your machine and make sure you have installed the necessary library.  

## How to load the nsysu-digits dataset
```python
import urllib
import numpy as np

base = r"https://github.com/SageLabTW/auto-grading/raw/master/nsysu-digits/"
urllib.request.urlretrieve(base + "X.csv", "nsysu-digits-X.csv")
urllib.request.urlretrieve(base + "y.csv", "nsysu-digits-y.csv")

X = np.genfromtxt('nsysu-digits-X.csv', dtype=int, delimiter=',')
y = np.genfromtxt('nsysu-digits-y.csv', dtype=int, delimiter=',')
```

## TODO list
- improve OCR
- test robustness of find_box and find_qr_box
- don't delete -1 when grading
- delete unauthorized data
- export data at the end of the semester