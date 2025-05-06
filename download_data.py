import os
import urllib.request
import tarfile
import shutil

if not os.path.exists('spamassassin_data'):
    os.makedirs('spamassassin_data')
    os.makedirs('spamassassin_data/spam')
    os.makedirs('spamassassin_data/ham')
print("Start downloading dataset...")

# Download regular mail (ham) and spam
"""
    https://spamassassin.apache.org/old/publiccorpus/
"""
ham_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
ham_file = "20030228_hard_ham.tar.bz2"
spam_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"
spam_file = "20050311_spam_2.tar.bz2"
urllib.request.urlretrieve(ham_url, ham_file)
urllib.request.urlretrieve(spam_url, spam_file)
# Unzip files
print("unzipping files...")
with tarfile.open(ham_file, "r:bz2") as tar:
    tar.extractall()
with tarfile.open(spam_file, "r:bz2") as tar:
    tar.extractall()
print("moving dataset to emlai_data")
ham_source = "easy_ham"
for filename in os.listdir(ham_source):
    if filename.startswith("cmds"):
        continue
    src = os.path.join(ham_source, filename)
    dst = os.path.join("spamassassin_data/ham", filename)
    shutil.copy2(src, dst)
spam_source = "spam"
for filename in os.listdir(spam_source):
    if filename.startswith("cmds"):
        continue
    src = os.path.join(spam_source, filename)
    dst = os.path.join("spamassassin_data/spam", filename)
    shutil.copy2(src, dst)
print("cleaning dates...")
shutil.rmtree(ham_source)
shutil.rmtree(spam_source)
os.remove(ham_file)
os.remove(spam_file)
print("dataset ready!")