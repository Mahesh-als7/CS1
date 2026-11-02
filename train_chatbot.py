from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
intents = json.loads((BASE_DIR / "intents.json").read_text(encoding="utf-8"))

texts, labels = [], []
for intent_name, content in intents.items():
    if intent_name == "fallback":
        continue
    for phrase in content["patterns"]:
        texts.append(phrase)
        labels.append(intent_name)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode")),
    ("classifier", MultinomialNB(alpha=0.35))
])

pipeline.fit(texts, labels)

# Save the fitted vectorizer and classifier together.
bundle = {
    "vectorizer": pipeline.named_steps["tfidf"],
    "classifier": pipeline.named_steps["classifier"]
}
joblib.dump(bundle, BASE_DIR / "chatbot_model.joblib")

print(f"Trained on {len(texts)} example phrases across {len(set(labels))} intents.")
print(f"Saved model to: {BASE_DIR / 'chatbot_model.joblib'}")

# A few quick sanity checks; these are examples, not a formal accuracy claim.
tests = [
    "hie there",
    "how are you doing today",
    "what exactly is phishing",
    "how can I make my password safer",
    "what is encryption used for",
]
for sample in tests:
    probs = pipeline.predict_proba([sample])[0]
    idx = probs.argmax()
    print(f"{sample!r} -> {pipeline.classes_[idx]} ({probs[idx]:.2f})")
