## SpamEmailKiller
### 1. Install
python3.11.0
```bash
python -m venv venv
pip install -r requirements.txt
```

### 2. run
#### 2.1 First time train and test
```
python main.py train
```
#### 2.2 Test Single Email
* From 
  * ```python main.py predict --email Get rich quick! --model nb```
* From text file
  * ```python main.py predict --email test_normal.txt --model nb```


version https://git-lfs.github.com/spec/v1
oid sha256:fa67f1660117e5adc27f3d40cb9c02f29d1fa7d8432a19779a7d7d4f49f5030c
size 538
