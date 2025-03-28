# Source Code for CS410 Final Project - Spam Email Detector
## 0. Setup
require: python 3.11.0
```bash
git clone git@github.com:KangweiZhu/SpamEmailKiller.git
cd SpamEmailKiller
python -m venv .venv
pip install -r requirements.txt
```

## 1. How to train and test
```bash
python main.py train
python main.py test
```

## 2. Predict a single email
- baseline
- nb
- svm

* Email Text using nb/svm/baseline. 
```bash
python main.py predict --model nb --email "Get rich quick! Special offer!"
```

* Load from text file
```bash
python main.py predict --model nb --email test_email.txt
```
```bash
python main.py predict --model nb --email test_normal.txt
```