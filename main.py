import os
from email_preprocessor import EmailPreprocessor
from spam_classifier import BaselineSpamClassifier, MLSpamClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from visualizer import SpamVisualizer
import argparse
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm


def parse_email_content(raw_email):
    lines = raw_email.split('\n')
    start = 0

    # Find the location of first empty line, which will be the postion of main body
    for i, line in enumerate(lines):
        if line.strip() == '':
            start = i + 1
            break
    body = '\n'.join(lines[start:])
    return body.lower().strip()


def load_data(data_dir):
    print(f"Loading data from {data_dir}")
    emails = []
    labels = []
    total_file_count = 0

    if 'test_data' in data_dir:
        # 测试数据使用 hard_ham
        label_types = ['spam', 'easy_ham']
    else:
        label_types = ['spam', 'ham']

    for label in label_types:
        label_dir = os.path.join(data_dir, label)
        file_count = 0
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content.strip()) > 0:
                    if 'email_data' in data_dir:
                        content = parse_email_content(content)
                    emails.append(content)
            if label == 'spam':
                labels.append(1)
            else:
                # 无论是 ham 还是 hard_ham 都是正常邮件
                labels.append(0)
            file_count += 1
        total_file_count += file_count
        print(f"Loaded {file_count} emails from {label}")
    print(f"Total emails: {len(emails)}")
    print(f"Spam emails: {sum(labels)}")
    print(f"Ham emails: {len(labels) - sum(labels)}")
    print(f"Spam percentage: {sum(labels) / len(labels):.2%}")
    return emails, labels


def generate_report(train_stats, test_results, output_dir='reports', is_drytest=False):
    visualizer = SpamVisualizer(output_dir)
    visualizer.plot_data_distributions(train_stats)
    visualizer.plot_performance_comparison(test_results)
    visualizer.plot_confusion_matrices(test_results)
    visualizer.plot_accuracy_table(test_results)
    visualizer.generate_text_report(train_stats, test_results, is_drytest)


def train_and_validate():
    print("Loading Enron dataset")
    enron_emails, enron_labels = load_data("enron_data")
    print("Loading SpamAssassin dataset")
    assain_emails, assasin_labels = load_data("email_data")
    all_emails = enron_emails + assain_emails
    all_labels = enron_labels + assasin_labels
    enron_sources = ['enron'] * len(enron_emails)
    assasin_sources = ['spamassassin'] * len(assain_emails)
    all_sources = enron_sources + assasin_sources
    x_train, x_val, y_train, y_val, src_train, src_val = train_test_split(
        all_emails, all_labels, all_sources,
        test_size=0.2,
        random_state=40,
        stratify=all_sources
    )
    print(f"trainset size: {len(x_train)}")
    print(f"validation set size: {len(x_val)}")
    print(f"validation set data from Enron: {src_val.count('enron')}")
    print(f"validation set data from assasin: {src_val.count('spamassassin')}")

    print('pre-process email train datas')
    preprocessor = EmailPreprocessor()
    train_data = []
    for i, email in enumerate(tqdm(x_train), 1):
        processed = preprocessor.preprocess_email(email)
        train_data.append(processed)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(x_train)} train emails")
    
    X_train_features = preprocessor.extract_features(train_data)

    print("pre-process email validation datas")
    val_data = []
    for i, email in enumerate(tqdm(x_val), 1):
        processed = preprocessor.preprocess_email(email)
        val_data.append(processed)
        if i % 1000 == 0:
            print(f"Processed {i}/{len(x_val)} validation emails")
    X_val_features = preprocessor.tfidf_vectorizer.transform(val_data)
    val_stats = {
        'enron_train': src_train.count('enron'),
        'spamassassin_train': src_train.count('spamassassin'),
        'enron_val': src_val.count('enron'),
        'spamassassin_val': src_val.count('spamassassin'),
        'val_ham': sum(1 for label in y_val if label == 0),
        'val_spam': sum(1 for label in y_val if label == 1),
        'total_ham': sum(1 for label in all_labels if label == 0),
        'total_spam': sum(1 for label in all_labels if label == 1)
    }
    test_results = {}
    print("Training all models")
    models = {
        'baseline': BaselineSpamClassifier(),
        'nb': MLSpamClassifier(classifier_type='nb'),
        'svm': MLSpamClassifier(classifier_type='svm')
    }

    for name, model in models.items():
        print(f"evaluating {name}")
        if name != 'baseline':
            model.train(X_train_features, y_train)
            y_pred = model.predict(X_val_features)
        else:
            # just use keyword matching
            y_pred = [model.predict(email) for email in x_val]
        test_results[name] = {
            'overall': {
                'accuracy': accuracy_score(y_val, y_pred),
                'confusion_matrix': confusion_matrix(y_val, y_pred)
            }
        }
        enron_mask = [i for i, src in enumerate(src_val) if src == 'enron']
        test_results[name]['enron'] = {
            'accuracy': accuracy_score(
                [y_val[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_val[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            )
        }
        spam_mask = [i for i, src in enumerate(src_val) if src == 'spamassassin']
        test_results[name]['spamassassin'] = {
            'accuracy': accuracy_score(
                [y_val[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_val[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            )
        }
    generate_report(val_stats, test_results)
    for name, model in models.items():
        with open(f"{name}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    with open("preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)

def final_test():
    print("Loading test data")
    test_emails, test_labels = load_data("test_data")
    print("Loading trained models")
    with open("preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
    models = {
        'baseline': pickle.load(open("baseline_model.pkl", 'rb')),
        'nb': pickle.load(open("nb_model.pkl", 'rb')),
        'svm': pickle.load(open("svm_model.pkl", 'rb'))
    }
    print("Processing test data")
    test_data = []
    for email in tqdm(test_emails):
        processed = preprocessor.preprocess_email(email)
        test_data.append(processed)
    X_test_features = preprocessor.tfidf_vectorizer.transform(test_data)

    test_stats = {
        'test_ham': sum(1 for label in test_labels if label == 0),
        'test_spam': sum(1 for label in test_labels if label == 1)
    }

    test_results = {}
    print("Evaluating models on test set")
    for name, model in models.items():
        if name != 'baseline':
            y_pred = model.predict(X_test_features)
        else:
            y_pred = [model.predict(email) for email in test_emails]
        test_results[name] = {
            'test': {
                'accuracy': accuracy_score(test_labels, y_pred),
                'confusion_matrix': confusion_matrix(test_labels, y_pred)
            }
        }
    visualizer = SpamVisualizer(output_dir='final_test_reports')
    visualizer.plot_test_performance(test_results)
    visualizer.plot_test_confusion_matrices(test_results)
    visualizer.generate_test_report(test_stats, test_results)

def interactive_predict(model_path, email_content):
    with open("preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    processed_content = preprocessor.preprocess_email(email_content)
    X = preprocessor.tfidf_vectorizer.transform([processed_content])

    if isinstance(model, BaselineSpamClassifier):
        prediction = model.predict(email_content)
        probability = [0.0, 1.0] if prediction == 1 else [1.0, 0.0]
    else:
        prediction = model.predict(X)[0]
        if hasattr(model.classifier, 'predict_proba'):
            probability = model.classifier.predict_proba(X)[0]
        else:
            probability = [0.0, 1.0] if prediction == 1 else [1.0, 0.0]

    return prediction, probability


def main():
    parser = argparse.ArgumentParser(description='Spam Email Detection System')
    parser.add_argument('command', choices=['train', 'validate', 'test'])
    parser.add_argument('--email', type=str, nargs='+')
    parser.add_argument('--model', type=str, default='nb', choices=['baseline', 'nb', 'svm'])
    args = parser.parse_args()

    if args.command == 'predict':
        if not args.email:
            parser.error("--email is required for predict command")
        model_path = f"{args.model}_model.pkl"
        email_content = None
        email_input = ' '.join(args.email)
        if os.path.exists(email_input):
            with open(email_input, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
        else:
            email_content = email_input
        prediction, probability = interactive_predict(model_path, email_content)
        result = "SPAM" if prediction == 1 else "HAM"
        spam_prob = probability[1]
        print(f"Classification: {result}")
        print(f"Spam Probability: {spam_prob:.2%}")
        print(f"Ham Probability: {(1 - spam_prob):.2%}")
    elif args.command == 'train':
        train_and_validate()
    elif args.command == 'test':
        final_test()

if __name__ == "__main__":
    main()
