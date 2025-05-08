import os
import sys
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import glob
import re
from datetime import datetime

# Ensure working directory is always the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

# Try to import train_and_test, final_test
try:
    from main import train_and_test, final_test
except Exception:
    train_and_test = None
    final_test = None

REPORTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'final_test_reports'))

app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path='')
CORS(app)

MODEL_FILENAMES = ['baseline_model.pkl', 'nb_model.pkl', 'svm_model.pkl', 'preprocessor.pkl']
MODEL_PATHS = {}
MODELS = {}
PREPROCESSOR = None
MODELS_READY = True
MISSING_FILES = []

# Initialize Gemini
GEMINI_API_KEY = None
GEMINI_MODEL = None

def init_gemini(api_key):
    global GEMINI_API_KEY, GEMINI_MODEL
    try:
        genai.configure(api_key=api_key)
        GEMINI_MODEL = genai.GenerativeModel('gemini-2.0-flash')
        GEMINI_API_KEY = api_key
        return True
    except Exception as e:
        print(f"Gemini initialization failed: {str(e)}")
        return False

def analyze_with_gemini(email_content):
    """Analyze email content using Gemini"""
    if GEMINI_MODEL is None:
        return "Gemini not initialized"
    
    try:
        prompt = f"""You are a Spam Email Filter. Analyze this email and determine if it's spam or ham (normal email).
        Provide your decision and a brief explanation.

        Email:
        {email_content}

        Respond in this exact format:
        Decision: [spam/ham]
        Reason: [your explanation]"""
        
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini analysis failed: {str(e)}"

# Helper to find model files robustly
def find_model_file(filename):
    # 1. Try project root
    path = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(path):
        return path
    # 2. Try current dir
    if os.path.exists(filename):
        return filename
    return None

# Check and load models
for fname in MODEL_FILENAMES:
    fpath = find_model_file(fname)
    if fpath is None:
        MODELS_READY = False
        MISSING_FILES.append(fname)
    else:
        MODEL_PATHS[fname] = fpath

if MODELS_READY:
    try:
        for name in ['baseline', 'nb', 'svm']:
            with open(MODEL_PATHS[f'{name}_model.pkl'], 'rb') as f:
                MODELS[name] = pickle.load(f)
        with open(MODEL_PATHS['preprocessor.pkl'], 'rb') as f:
            PREPROCESSOR = pickle.load(f)
    except Exception as e:
        MODELS_READY = False
        MISSING_FILES.append(str(e))

# Helper for error response
MODEL_NOT_READY_MSG = {
    'error': 'Model files missing or not ready. Please run training first.',
    'missing_files': MISSING_FILES
}

@app.route('/predict', methods=['POST'])
def predict():
    # Always check model files on every request
    missing = []
    for fname in MODEL_FILENAMES:
        fpath = find_model_file(fname)
        if fpath is None or not os.path.exists(fpath):
            missing.append(fname)
    if missing:
        return jsonify({'error': 'Model files missing or not ready. Please run training first.', 'missing_files': missing}), 503

    # Reload models and preprocessor on every request
    global MODELS, PREPROCESSOR
    MODELS = {}
    for name in ['baseline', 'nb', 'svm']:
        with open(find_model_file(f'{name}_model.pkl'), 'rb') as f:
            MODELS[name] = pickle.load(f)
    with open(find_model_file('preprocessor.pkl'), 'rb') as f:
        PREPROCESSOR = pickle.load(f)

    data = request.json
    email_content = data.get('email')
    model_name = data.get('model', 'nb')
    use_gemini = data.get('use_gemini', False)
    gemini_api_key = data.get('gemini_api_key')
    
    if not email_content:
        return jsonify({'error': 'No email content provided'}), 400
    
    # 如果启用了Gemini分析
    if use_gemini:
        if not gemini_api_key:
            return jsonify({'error': 'Gemini API key is required'}), 400
        
        # 如果API key改变，重新初始化
        if gemini_api_key != GEMINI_API_KEY:
            if not init_gemini(gemini_api_key):
                return jsonify({'error': 'Failed to initialize Gemini'}), 500
        
        gemini_analysis = analyze_with_gemini(email_content)
        return jsonify({
            'result': 'SPAM' if 'Decision: spam' in gemini_analysis else 'HAM',
            'analysis': gemini_analysis
        })
    
    # 使用传统模型
    if model_name not in MODELS:
        return jsonify({'error': 'Invalid model name'}), 400
    model = MODELS[model_name]
    if model_name == 'baseline':
        pred = model.predict(email_content)
        prob = float(pred)
    else:
        processed = PREPROCESSOR.preprocess_email(email_content)
        X = PREPROCESSOR.tfidf_vectorizer.transform([processed])
        pred = model.predict(X)[0]
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'predict_proba'):
            prob = model.classifier.predict_proba(X)[0][1]
        else:
            prob = float(pred)
    result = "SPAM" if pred == 1 else "HAM"
    return jsonify({
        'result': result,
        'spam_probability': prob,
        'ham_probability': 1 - prob
    })

@app.route('/train', methods=['POST'])
def train():
    if train_and_test is None:
        return jsonify({'error': 'Training function not available.'}), 500
    try:
        train_and_test()
        return jsonify({'status': 'success', 'message': 'Training completed'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/test', methods=['POST'])
def test():
    if final_test is None:
        return jsonify({'error': 'Test function not available.'}), 500
    missing = []
    for fname in MODEL_FILENAMES:
        fpath = find_model_file(fname)
        if fpath is None or not os.path.exists(fpath):
            missing.append(fname)
    if missing:
        return jsonify({'error': 'Model files missing or not ready. Please run training first.', 'missing_files': missing}), 503
    try:
        final_test()
        return jsonify({'status': 'success', 'message': 'Testing completed'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    # Always check model files on every request
    missing = []
    for fname in MODEL_FILENAMES:
        fpath = find_model_file(fname)
        if fpath is None or not os.path.exists(fpath):
            missing.append(fname)
    models_ready = (len(missing) == 0)
    return jsonify({
        'models_ready': models_ready,
        'missing_files': missing
    })

def get_latest_report_files():
    """获取最新的测试报告文件"""
    report_files = glob.glob(os.path.join(REPORTS_DIR, 'test_report_*.txt'))
    if not report_files:
        return None, None, None, None, None, None
    latest_report = max(report_files, key=lambda x: os.path.getctime(x))
    timestamp = re.search(r'test_report_(\d{8}_\d{6})\.txt', latest_report).group(1)
    performance_img = os.path.join(REPORTS_DIR, f'test_performance_{timestamp}.png')
    baseline_cm = os.path.join(REPORTS_DIR, f'baseline_test_confusion_matrix_{timestamp}.png')
    nb_cm = os.path.join(REPORTS_DIR, f'nb_test_confusion_matrix_{timestamp}.png')
    svm_cm = os.path.join(REPORTS_DIR, f'svm_test_confusion_matrix_{timestamp}.png')
    return latest_report, performance_img, baseline_cm, nb_cm, svm_cm, timestamp

@app.route('/latest_report', methods=['GET'])
def get_latest_report():
    try:
        # 获取最新的测试报告文件
        report_file, performance_img, baseline_cm, nb_cm, svm_cm, timestamp = get_latest_report_files()
        if not report_file:
            return jsonify({
                'error': 'No test report found. Please run the test first.'
            }), 404

        # 读取报告内容
        with open(report_file, 'r') as f:
            content = f.read()

        # 解析报告内容
        report_data = {
            'timestamp': None,
            'models': {},
            'images': {
                'performance': None,
                'baseline_cm': None,
                'nb_cm': None,
                'svm_cm': None
            }
        }

        # 提取时间戳
        timestamp_match = re.search(r'Test Time: (.*?)\n', content)
        if timestamp_match:
            report_data['timestamp'] = timestamp_match.group(1)

        # 提取每个模型的结果
        model_sections = re.split(r'\n(?=Model: )', content)
        for section in model_sections:
            if not section.strip():
                continue

            model_match = re.search(r'Model: (.*?)\n', section)
            if not model_match:
                continue

            model_name = model_match.group(1)
            model_data = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'confusion_matrix': None
            }

            # 提取各项指标
            accuracy_match = re.search(r'Accuracy: ([\d.]+)', section)
            if accuracy_match:
                model_data['accuracy'] = float(accuracy_match.group(1))

            precision_match = re.search(r'Precision: ([\d.]+)', section)
            if precision_match:
                model_data['precision'] = float(precision_match.group(1))

            recall_match = re.search(r'Recall: ([\d.]+)', section)
            if recall_match:
                model_data['recall'] = float(recall_match.group(1))

            f1_match = re.search(r'F1 Score: ([\d.]+)', section)
            if f1_match:
                model_data['f1_score'] = float(f1_match.group(1))

            # 提取混淆矩阵
            cm_match = re.search(r'Confusion Matrix:\n(.*?)(?=\n\n|\Z)', section, re.DOTALL)
            if cm_match:
                model_data['confusion_matrix'] = cm_match.group(1).strip()

            report_data['models'][model_name] = model_data

        # 添加图片路径
        if os.path.exists(performance_img):
            report_data['images']['performance'] = f'http://localhost:5002/final_test_reports/test_performance_{timestamp}.png'
        if os.path.exists(baseline_cm):
            report_data['images']['baseline_cm'] = f'http://localhost:5002/final_test_reports/baseline_test_confusion_matrix_{timestamp}.png'
        if os.path.exists(nb_cm):
            report_data['images']['nb_cm'] = f'http://localhost:5002/final_test_reports/nb_test_confusion_matrix_{timestamp}.png'
        if os.path.exists(svm_cm):
            report_data['images']['svm_cm'] = f'http://localhost:5002/final_test_reports/svm_test_confusion_matrix_{timestamp}.png'

        return jsonify(report_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False) 