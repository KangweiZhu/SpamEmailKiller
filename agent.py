import logging
import os
from asyncio import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
import logging
import os
from datetime import datetime

import requests


class Email:
    def __init__(self, spam=False, content=None, file_path=None, source=None):
        """
        Constructor

        :param spam: True if it is a spam email, otherwise False
        :param content: Email content. Might be proprocessed or not preprocessed
        :param file_path: path of the file
        :param source: Spamassassin or Enron.
        """
        self.spam = spam
        self.content = content
        self.file_path = file_path
        self.source = source


def llm_classifier(email: Email, model='qwen2.5:0.5b'):
    """
    Input:  Spam/Ham email content
    Output: Spam or Ham

    :param email_text: Email text
    :param model: by default qwen2.5:0.5b
    :return: Response 'Ham or Spam'
    """
    resp = requests.request(
        method='POST',
        url='http://localhost:11434/api/generate',
        json={
            "model": model,
            "prompt": f"""
                You are a Spam Email Filter. Please judge the email below is a normal email(ham) or spam email(spam)
                    
                === Begin of email ===
                {email.content}
                === End of email ===
                
                If it is a spam email, please only respond me with a single word: spam. If it is a ham email, please only 
                respond me with a single word: ham. 
            """,
            "stream": False
        }
    )
    return resp


def load_emails(base_dir):
    """
    Load email from a data dir.

    :param base_dir:
    :return:
    """
    data: list[Email] = []
    for label_name in ['spam', 'ham']:
        label = True if label_name == 'spam' else False
        data_dir = os.path.join(base_dir, label_name)
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        data.append(
                            Email(
                                content=file_content,
                                file_path=file_path,
                                spam=label,
                                source=base_dir
                            )
                        )
                except Exception as e:
                    print(f'Failed to read from {file_path}.')
    return data

"""
    Feed LLM with raw email data in each dataset

    :param email_data:
    :param max_threads: Control concurrent request
    :param max_samples: Limit the number of emails that ollama could handle
    :return: None
    """
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_classification(email_data, max_threads=4, max_samples=30):
    email_data = email_data[:max_samples]
    success_count = 0
    error_count = 0
    correct_count = 0
    wrong_count = 0

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(llm_classifier, email): (i+1, email) for i, email in enumerate(email_data)}

        for future in as_completed(futures):
            index, email = futures[future]
            try:
                resp = future.result()
                result = resp.json().get("response", "").strip().lower()
            except Exception as e:
                error_count += 1
                logging.info(f"index={index},source={email.source},file={email.file_path},error={str(e)}")
                continue

            if result not in ['spam', 'ham']:
                error_count += 1
                logging.info(f"index={index},source={email.source},file={email.file_path},error=invalid_response:{result}")
                continue

            pred_label = True if result == "spam" else False
            true_label = "spam" if email.spam else "ham"
            predicted = result
            is_correct = (pred_label == email.spam)

            success_count += 1
            if is_correct:
                correct_count += 1
                result_status = "correct"
            else:
                wrong_count += 1
                result_status = "wrong"

            logging.info(f"index={index},source={email.source},file={email.file_path},true={true_label},pred={predicted},result={result_status}")

    logging.info(f"summary,total={len(email_data)},success={success_count},wrong={wrong_count},error={error_count},accuracy={correct_count / success_count:.4f}" if success_count else "summary,error_only")


def setup_logger():
    os.makedirs("log", exist_ok=True)
    log_filename = datetime.now().strftime("log/classification_%Y-%m-%d_%H-%M-%S.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers (to prevent rewriting)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logging.info(f"start_log_file={log_filename}")


setup_logger()
enron_data = load_emails('enron_data')
spamassassin_data = load_emails('spamassassin_data')
all_data = enron_data + spamassassin_data
run_classification(all_data, max_threads=6, max_samples=100)