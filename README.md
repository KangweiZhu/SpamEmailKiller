# Spam Email Detector
## 1. How to train and test
```bash
python main.py train
```

## 2. Predict a single email
- baseline
- nb
- svm

* Email Text
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