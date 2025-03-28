import os
import urllib.request
import ssl
import tarfile
import shutil
from tqdm import tqdm

os.makedirs('temp', exist_ok=True)
os.makedirs('enron_data/spam', exist_ok=True)
os.makedirs('enron_data/ham', exist_ok=True)
base_url = "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/"

context = ssl._create_unverified_context()

for i in range(1, 7):
    filename = f"enron{i}.tar.gz"
    url = f"{base_url}/{filename}"
    tar_path = os.path.join('temp', filename)

    print(f"\nDownloading {url}...")

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        with urllib.request.urlopen(url, context=context) as response:
            with open(tar_path, 'wb') as out_file:
                content_length = response.headers.get('content-length')
                if content_length:
                    t.total = int(content_length)
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    t.update(len(chunk))

    print(f"Extracting {filename}")
    with tarfile.open(tar_path) as tar:
        tar.extractall('temp')

    enron_dir = os.path.join('temp', f'enron{i}')
    if os.path.exists(enron_dir):
        spam_dir = os.path.join(enron_dir, 'spam')
        if os.path.exists(spam_dir):
            for f in os.listdir(spam_dir):
                src = os.path.join(spam_dir, f)
                dst = os.path.join('enron_data/spam', f"enron{i}_{f}")
                shutil.copy2(src, dst)

        ham_dir = os.path.join(enron_dir, 'ham')
        if os.path.exists(ham_dir):
            for f in os.listdir(ham_dir):
                src = os.path.join(ham_dir, f)
                dst = os.path.join('enron_data/ham', f"enron{i}_{f}")
                shutil.copy2(src, dst)

spam_count = len(os.listdir('enron_data/spam'))
ham_count = len(os.listdir('enron_data/ham'))
print("\nDataset preparation completed!")
print(f"Total emails: {spam_count + ham_count}")
print(f"Spam emails: {spam_count}")
print(f"Ham emails: {ham_count}")
print("\nCleaning up temporary files...")
shutil.rmtree('temp', ignore_errors=True)
