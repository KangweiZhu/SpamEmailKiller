# Source Code for CS410 Final Project - Spam Email Detector
## 0. Setup
require: python 3.11.0
```bash
git clone git@github.com:KangweiZhu/SpamEmailKiller.git
cd SpamEmailKiller
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. How to train and test
```bash
python main.py train
python main.py test
```

## 2. Predict a single email 📧

### 🔍 Usage
```bash
python main.py predict --model <model_name> --email <path_to_your_email.txt>
```

- `<model_name>` must be one of: `nb`, `svm`, or `baseline`  
- `<path_to_your_email.txt>` can be:
  - A **quoted email string**  
    e.g., `"Get rich quick! Special offer!"`
  - Or a **path to a plain text file** containing the email content  
    e.g., `test_email.txt`

---

### 💡 Examples

**Text:**
```bash
python main.py predict --model swm --email "Exclusive deal just for you!"
```

**Email from file:**
```bash
python main.py predict --model nb --email test_email.txt
```

**Using the baseline model:**
```bash
python main.py predict --model baseline --email "Meeting scheduled at 2pm tomorrow."
```

## 3. Convert `.eml` Email to Plain Text 📄

We wrote a convertor can convert simple email to text. To convert a raw email file (`.eml`) into a plain text file suitable for prediction, use the following script:

### 🔄 Usage
```bash
python process.py <path_to_eml> <path_to_txt>
```

- `<path_to_eml>`: Path to the raw email file (e.g., `emails/spam_email.eml`)
- `<path_to_txt>`: Path where the processed plain text should be saved (e.g., `emails/spam_email.txt`)

---

### 🔍 Example
```bash
python process.py test_eml.eml test_eml.txt
```

You can then use the resulting `.txt` file for prediction.

## 4. 🖥️ GUI: Email Classification Tool
Our GUI can be run simply by:
```bash
python gui.py
```
> 💡 Tip: If you select a `.eml` file, it will be automatically processed into a `.txt` file before prediction.