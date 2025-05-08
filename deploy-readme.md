​	

> Due to the nature of our project's delivery, it is hard to run the project by one-click or running one single script. Thus we recommend to just check our project demonstration video in the final project presentation Google drive.

### 1. Make Sure the system has `nodejs` and `npm`

```
node -v
npm -v
```



### 2. Make Sure the system has `python3.11.0`

```
pyenv global 3.11.0
python --version
```



### 3. Make sure no processes running on port `5173` and `5002`

```
lsof -i :5173
kill -9 $(lsof -t -i :5173)
```

```
lsof -i :5002
kill -9 $(lsof -t -i :5002)
```



### 4. Create python venv at project root directory

```
python -m venv .venv
source .venv/bin/activate
```



### 5. Install dependencies

```
pip install -r requirements.txt
```

If shows dependency missing, then simply re-install the missing one could solve the issue



### 6. Start backend

```
cd spamemailkiller-app/backend
python spam_api.py
# or nohup python spam_api.py &
```



### 7. Start frontend 

```
cd ../frontend
rm -rf node_modules # only run if failed to start the process
npm install
npm run dev
```







