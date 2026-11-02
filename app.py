from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import json
import random
import re
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "chatbot_model.joblib"
INTENTS_PATH = BASE_DIR / "intents.json"
DASHBOARD_PATH = BASE_DIR / "CyberShield_Redesigned_With_Chatbot.html"

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Local educational demo only.

model_bundle = None

def load_model():
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
    else:
        model_bundle = None

@app.get("/")
def home():
    if DASHBOARD_PATH.exists():
        return send_file(DASHBOARD_PATH)
    return "CyberShield dashboard file not found. Keep the HTML file beside app.py.", 404

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_bundle is not None})

@app.post("/chat")
def chat():
    if model_bundle is None:
        return jsonify({"error": "Chatbot model is not trained yet. Run python train_chatbot.py first."}), 503

    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 500:
        return jsonify({"error": "Please keep messages under 500 characters."}), 400

    vectorizer = model_bundle["vectorizer"]
    classifier = model_bundle["classifier"]
    intents = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))

    probabilities = classifier.predict_proba(vectorizer.transform([message]))[0]
    best_index = probabilities.argmax()
    intent = classifier.classes_[best_index]
    confidence = float(probabilities[best_index])

    # A confidence threshold prevents uncertain guesses from being presented as certain.
    if confidence < 0.28 or intent == "fallback":
        intent = "fallback"

    response = random.choice(intents[intent]["responses"])
    return jsonify({
        "response": response,
        "intent": intent,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    load_model()
    print("CyberShield local chatbot: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)
