"""
text_urgency_model.py

The second AI technique in this project (alongside A* search in routing.py):
a small TRAINED text classifier that reads the free-text `description` field
of a report and estimates how much it should shift the priority score, based
on learned word patterns — not a hand-written keyword list.

Pipeline (worth walking through in a viva):
  1. Feature extraction: TF-IDF over the description text (unigrams + bigrams).
  2. Model: multinomial logistic regression, trained on a labeled seed set of
     representative disaster-report phrases (TRAINING_DATA below).
  3. Prediction: for a new description, predict an urgency class in
     {-1, 0, 1, 2} and convert that to a score boost/penalty.
  4. Explainability: for the predicted class, rank the words actually present
     in *this* description by their learned coefficient, so you can show
     *why* the model reacted the way it did (important for defending an AI
     component that isn't just if/else rules).

The model trains once automatically (if urgency_model.joblib isn't present)
and is reused after that — no separate training step required to run the app.
Delete urgency_model.joblib to force retraining (e.g. after editing
TRAINING_DATA).

This is intentionally a small seed dataset. The point of a course prototype
is to demonstrate the *pipeline* — train -> save -> load -> predict -> explain
— which is identical whether trained on 30 examples or 30,000. Swapping in a
bigger/real dataset later means only changing TRAINING_DATA.
"""

import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "urgency_model.joblib"

# label: -1 = lowers urgency, 0 = neutral, 1 = raises urgency, 2 = raises it sharply
TRAINING_DATA = [
    ("trapped under rubble, cannot move", 2),
    ("person unconscious, not breathing", 2),
    ("child trapped on rooftop, water rising", 2),
    ("multiple people bleeding heavily", 2),
    ("building collapsed, people still inside", 2),
    ("fire spreading fast, families still inside", 2),
    ("elderly person stuck on second floor, flooding", 2),
    ("baby separated from parents in the crowd", 2),
    ("gas leak near shelter, risk of explosion", 2),
    ("victim not responding to voice", 2),
    ("severe bleeding, needs immediate evacuation", 2),
    ("stranded on roof, water still rising fast", 2),
    ("no pulse detected, cpr in progress", 2),
    ("children trapped inside collapsed structure", 2),

    ("road blocked by fallen tree", 1),
    ("bridge partially collapsed, single lane passable", 1),
    ("hospital reporting reduced bed capacity", 1),
    ("landslide blocking main access road", 1),
    ("power lines down across the street", 1),
    ("ambulance access blocked by debris", 1),
    ("shelter running low on supplies", 1),
    ("injured person, conscious but in pain", 1),
    ("main highway flooded, detour needed", 1),
    ("hospital requesting additional staff", 1),

    ("minor delay expected, road partly clear", 0),
    ("hospital status update, no change", 0),
    ("routine check, no injuries reported", 0),
    ("area currently stable", 0),
    ("waiting for further instructions", 0),
    ("small crowd gathering, no danger noted", 0),
    ("generator running normally", 0),

    ("minor scratch, walking wounded, refused treatment", -1),
    ("false alarm, area already cleared", -1),
    ("road reopened, traffic flowing normally", -1),
    ("hospital reports surplus capacity", -1),
    ("situation resolved, no further action needed", -1),
    ("light traffic delay only", -1),
    ("all residents accounted for, no injuries", -1),
]

_vectorizer = None
_model = None

# How much each predicted class shifts the rule-based priority score.
CLASS_TO_BOOST = {-1: -1.0, 0: 0.0, 1: 1.5, 2: 3.0}


def _train_and_save():
    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
    return vectorizer, model


def _load_model():
    global _vectorizer, _model
    if _vectorizer is not None and _model is not None:
        return _vectorizer, _model

    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        _vectorizer, _model = saved["vectorizer"], saved["model"]
    else:
        _vectorizer, _model = _train_and_save()

    return _vectorizer, _model


@dataclass
class UrgencyPrediction:
    boost: float            # amount to add to the priority_score
    predicted_class: int    # -1, 0, 1, or 2
    top_terms: list         # words/phrases in this text that drove the prediction
    confidence: float       # model's probability for its top class


def predict_urgency(description: str) -> UrgencyPrediction:
    """
    Run the trained text classifier over a report's free-text description.
    Returns a score boost/penalty plus which words drove the prediction
    (for the dashboard and for explaining the model in a viva).
    """
    if not description or not description.strip():
        return UrgencyPrediction(boost=0.0, predicted_class=0, top_terms=[], confidence=0.0)

    vectorizer, model = _load_model()
    X = vectorizer.transform([description])

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    pred_idx = int(np.argmax(proba))
    predicted_class = int(classes[pred_idx])
    confidence = float(proba[pred_idx])

    # Explainability: of the words actually present in THIS description,
    # which ones have the strongest learned weight toward the predicted class.
    feature_names = vectorizer.get_feature_names_out()
    present_idx = X.nonzero()[1]
    class_row = list(classes).index(predicted_class)
    weights = model.coef_[class_row]
    scored_terms = sorted(
        ((feature_names[i], weights[i]) for i in present_idx),
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    top_terms = [term for term, w in scored_terms[:3] if abs(w) > 0.01]

    boost = CLASS_TO_BOOST.get(predicted_class, 0.0)
    return UrgencyPrediction(
        boost=boost,
        predicted_class=predicted_class,
        top_terms=top_terms,
        confidence=round(confidence, 3),
    )
