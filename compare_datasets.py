import os
import shutil
from tqdm import tqdm

def load_emails(directory, ham_type='ham'):
    emails = {'spam': {}, ham_type: {}}
    for label in emails.keys():
        label_dir = os.path.join(directory, label)
        print(f"Loading {label} from {label_dir}")     
        for filename in tqdm(os.listdir(label_dir)):
            file_path = os.path.join(label_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    emails[label][content] = filename
            except Exception as e:
                print(f"Error reading {file_path}: {e}")          
    return emails

backup_dir = "test_data_backup"
if not os.path.exists(backup_dir):
    shutil.copytree("test_data", backup_dir)
    print(f"Created backup at {backup_dir}")
print("Loading spamassassin_data dataset")
email_data = load_emails('spamassassin_data', 'ham')
print("Loading test_data dataset")
test_data = load_emails('test_data', 'easy_ham')


files_to_remove = []
print("Checking spam duplicates")
for content, test_filename in test_data['spam'].items():
    if content in email_data['spam'].keys():
        files_to_remove.append(os.path.join("test_data/spam", test_filename))
print("Checking ham duplicates")
for content, test_filename in test_data['easy_ham'].items():
    if content in email_data['ham'].keys():
        files_to_remove.append(os.path.join("test_data/easy_ham", test_filename))

print("==== Before Removal ===")
print(f"Test data spam: {len(test_data['spam'])}")
print(f"Test data ham: {len(test_data['easy_ham'])}")
print(f"Duplicates to remove: {len(files_to_remove)}")
for file_path in files_to_remove:
    try:
        os.remove(file_path)
        print(f"Removed: {file_path}")
    except Exception as e:
        print(f"Error removing {file_path}: {e}")


remaining_spam = len([f for f in os.listdir("test_data/spam") if f != '.DS_Store'])
remaining_ham = len([f for f in os.listdir("test_data/easy_ham") if f != '.DS_Store'])
print("===\After Removal ===")
print(f"Remaining test data spam: {remaining_spam}")
print(f"Remaining test data ham: {remaining_ham}")
print(f"Backup of original test data is saved at: {backup_dir}")
