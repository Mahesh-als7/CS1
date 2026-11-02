# CyberShield local ML chatbot

This is a small, locally running intent-classification chatbot. It does **not** call Gemini, OpenAI, or any external AI API.

## How it works
1. `intents.json` contains example user phrases and prepared responses for each intent.
2. `train_chatbot.py` trains a TF-IDF + Multinomial Naive Bayes classifier from those example phrases.
3. `app.py` loads the trained model and exposes a local `/chat` endpoint.
4. The redesigned CyberShield dashboard sends chat messages to that local Python app.

The model predicts the user's intent. The app then selects one of the prepared responses for that intent. This is machine learning, but it is **not** a generative chatbot and it cannot reliably answer arbitrary questions.

## Setup (Windows)
This setup is compatible with your Python 3.14 installation; scikit-learn 1.8 and newer support Python 3.14. Open Command Prompt in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py train_chatbot.py
py app.py
```

Then open this address in your browser:

`http://127.0.0.1:5000`

Keep the dashboard HTML, `app.py`, `intents.json`, and `chatbot_model.joblib` in the same folder.

## Stop the app
Press `Ctrl + C` in the Command Prompt window.

## Add more training examples
Edit `intents.json`. Add natural ways people might express an intent under its `patterns` list. Add suitable responses under `responses`. Then retrain:

```bash
py train_chatbot.py
```

Restart `app.py` after retraining.

## Notes
- Internet is needed only to install the Python packages the first time. After installation and model training, the chatbot runs locally.
- No API key is required.
- This demo binds to `127.0.0.1` so it is available only on the same computer.
- The ML model is separate from CyberShield's phishing-detection model.
- Chatbot predictions can be wrong. It is a classroom demonstration, not a security product.
