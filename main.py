import os
from email_preprocessor import EmailPreprocessor
from spam_classifier import BaselineSpamClassifier, MLSpamClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from visualizer import SpamVisualizer
import argparse

# http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html
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
    print(f"Loading data from  {data_dir}")
    emails = []
    labels = []
    
    for label in ['spam', 'ham']:
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
                        labels.append(0)
                    file_count += 1
                    # print(f"{label} filenumL: {file_count}:")
                    # print(content[:100])
            
        
        print(f"Loaded {file_count} emails from {label}")

    print(f"total emails: {len(emails)}")
    print(f"spam emails: {sum(labels)}")
    print(f"ham emails: {len(labels) - sum(labels)}")
    print(f"spam email percentage: {sum(labels)/len(labels):.2%}") # Because we label each spam email as 1, thus we can use sum

    # lengths = [len(email.split()) for email in emails]
    # print(f"邮件最少单词含量: {min(lengths)}")
    # print(f"最大: {max(lengths)}")
    # print(f"avg: {sum(lengths)/len(lengths):.2f}")
    
    return emails, labels

def generate_report(train_stats, test_results, output_dir='reports', is_drytest=False):
    visualizer = SpamVisualizer(output_dir)
    visualizer.plot_data_distributions(train_stats)
    visualizer.plot_performance_comparison(test_results)
    visualizer.plot_confusion_matrices(test_results)
    visualizer.plot_accuracy_table(test_results)
    visualizer.generate_text_report(train_stats, test_results, is_drytest)

def train_and_test():
    print("Loading Enron dataset")
    enron_emails, enron_labels = load_data("enron_data")
    print("Loading SpamAssassin dataset")
    assain_emails, assasin_labels = load_data("email_data")

    all_emails = enron_emails + assain_emails
    all_labels = enron_labels + assasin_labels

    enron_sources = ['enron'] * len(enron_emails)
    assasin_sources = ['spamassassin'] * len(assain_emails)
    all_sources = enron_sources + assasin_sources

    x_train, x_test, y_train, y_test, src_train, src_test = train_test_split(
        all_emails, all_labels, all_sources, 
        test_size=0.2, 
        random_state=40,
        stratify=all_sources
    )

    print(f"trainset size: {len(x_train)}")
    print(f"testet size size: {len(x_test)}")
    print(f"testsset data from Enron: {src_test.count('enron')}")
    print(f"testset data from assasin: {src_test.count('spamassassin')}")
    
    print('pre-process email train datas')
    train_data = []
        preprocessor = EmailPreprocessor()
    for i, email in enumerate(x_train, 1):
        processed = preprocessor.preprocess_email(email)
        train_data.append(processed)
        if i % 100 == 0:
            print(f"Processed {i}/{len(x_train)} train email")
    X_train_features = preprocessor.extract_features(train_data)

    print("pre-process email test datas")
    test_data = []
    for i, email in enumerate(x_test, 1):
            processed = preprocessor.preprocess_email(email)
        test_data.append(processed)
        if i % 100 == 0:
            print(f"processed {i}/{len(x_test)} test email")
    X_test_features = preprocessor.tfidf_vectorizer.transform(test_data)

    train_stats = {
        'enron_train': src_train.count('enron'),
        'spamassassin_train': src_train.count('spamassassin'),
        'enron_test': src_test.count('enron'),
        'spamassassin_test': src_test.count('spamassassin'),
        'test_ham': sum(1 for label in y_test if label == 0),
        'test_spam': sum(1 for label in y_test if label == 1),
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
            y_pred = model.predict(X_test_features)
        else:
            # just use keyword matching
            y_pred = [model.predict(email) for email in x_test]
        
        test_results[name] = {
            'overall': {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        }

        enron_mask = [i for i, src in enumerate(src_test) if src == 'enron']
        test_results[name]['enron'] = {
            'accuracy': accuracy_score(
                [y_test[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_test[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            )
        }

        spam_mask = [i for i, src in enumerate(src_test) if src == 'spamassassin']
        test_results[name]['spamassassin'] = {
            'accuracy': accuracy_score(
                [y_test[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_test[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            )
        }
    generate_report(train_stats, test_results)

    for name, model in models.items():
        with open(f"{name}_model.pkl", 'wb') as f:
            pickle.dump(model, f)

    with open("preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)

def drytest():
    print("Loading models")
    with open("preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)

    models = {
        'baseline': pickle.load(open("baseline_model.pkl", 'rb')),
        'nb': pickle.load(open("nb_model.pkl", 'rb')),
        'svm': pickle.load(open("svm_model.pkl", 'rb'))
    }

    print("Loading Enron dataset")
    enron_emails, enron_labels = load_data("enron_data")
    print("Loading SpamAssassin dataset")
    assain_emails, assasin_labels = load_data("email_data")

    all_emails = enron_emails + assain_emails
    all_labels = enron_labels + assasin_labels

    enron_sources = ['enron'] * len(enron_emails)
    assasin_sources = ['spamassassin'] * len(assain_emails)
    all_sources = enron_sources + assasin_sources

    _, x_test, _, y_test, _, src_test = train_test_split(
        all_emails, all_labels, all_sources, 
        test_size=0.2, 
        random_state=40,
        stratify=all_sources
    )

    print(f"Total test samples: {len(x_test)}")
    print(f"From Enron: {src_test.count('enron')}")
    print(f"From SpamAssassin: {src_test.count('spamassassin')}")

    test_data = []
    for i, email in enumerate(x_test, 1):
        processed = preprocessor.preprocess_email(email)
        test_data.append(processed)
        if i % 100 == 0:
            print(f"Processed {i}/{len(x_test)} test emails")
    
    X_test_features = preprocessor.tfidf_vectorizer.transform(test_data)

    train_stats = {
        'enron_train': 0,
        'spamassassin_train': 0,
        'enron_test': src_test.count('enron'),
        'spamassassin_test': src_test.count('spamassassin'),
        'test_ham': sum(1 for label in y_test if label == 0),
        'test_spam': sum(1 for label in y_test if label == 1),
        'total_ham': sum(1 for label in all_labels if label == 0),
        'total_spam': sum(1 for label in all_labels if label == 1)
    }

    test_results = {}
    print("Evaluating models")
    
    for name, model in models.items():
        print(f"Testing {name} model")
        if name != 'baseline':
            y_pred = model.predict(X_test_features)
        else:
            y_pred = [model.predict(email) for email in x_test]
        
        test_results[name] = {
            'overall': {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        }

        enron_mask = [i for i, src in enumerate(src_test) if src == 'enron']
        test_results[name]['enron'] = {
            'accuracy': accuracy_score(
                [y_test[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_test[i] for i in enron_mask],
                [y_pred[i] for i in enron_mask]
            )
        }

        spam_mask = [i for i, src in enumerate(src_test) if src == 'spamassassin']
        test_results[name]['spamassassin'] = {
            'accuracy': accuracy_score(
                [y_test[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            ),
            'confusion_matrix': confusion_matrix(
                [y_test[i] for i in spam_mask],
                [y_pred[i] for i in spam_mask]
            )
        }

    print("Generating test report")
    generate_report(train_stats, test_results, output_dir='drytest_reports', is_drytest=True)

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
    parser.add_argument('command', choices=['train', 'drytest', 'predict'])
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
        print(f"Ham Probability: {(1-spam_prob):.2%}")

    elif args.command == 'train':
        train_and_test()
    
    elif args.command == 'drytest':
        drytest()

if __name__ == "__main__":
    main()
