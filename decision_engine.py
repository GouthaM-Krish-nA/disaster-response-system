"""
decision_engine.py

This is the "AI decision engine" stage of the pipeline. It takes a raw report
and produces a priority score + label + recommended action, combining TWO
AI techniques:

  1. A rule-based structured scorer (below) over type/severity/duplicates —
     deliberately simple and explainable, easy to defend in a viva.
  2. A trained ML text classifier (text_urgency_model.py) that reads the
     free-text `description` and predicts an urgency boost/penalty from
     learned word patterns (TF-IDF + logistic regression), with the specific
     words that drove its prediction surfaced for explainability.

The two are combined by addition: the rule-based part handles the fields a
form always has (type, severity), and the ML part extracts extra signal from
whatever free text a witness typed, which the rules alone can't see.
"""

from dataclasses import dataclass, field

from text_urgency_model import predict_urgency

# How urgent each report type is, all else being equal.
# victim_triage > road_block > hospital_status
TYPE_WEIGHT = {
    "victim_triage": 3.0,
    "road_block": 2.0,
    "hospital_status": 1.5,
}

RECOMMENDED_ACTION = {
    "victim_triage": "Dispatch nearest rescue team",
    "road_block": "Reroute responders, notify traffic control",
    "hospital_status": "Update hospital capacity registry",
}


@dataclass
class Decision:
    priority_score: float
    priority_label: str
    recommended_action: str
    text_urgency_class: int = 0        # -1, 0, 1, 2 — from the ML text classifier
    text_urgency_confidence: float = 0.0
    text_urgency_terms: list = field(default_factory=list)


def score_report(report_type: str, severity: int, duplicate_count: int = 0,
                  description: str = "") -> Decision:
    """
    Compute a priority score for an incoming report.

    Args:
        report_type: one of "victim_triage", "road_block", "hospital_status"
        severity: 1 (minor) to 5 (critical), as reported by the source
        duplicate_count: how many similar reports (same type + nearby location)
                         have come in recently. More duplicates = more confidence
                         this is real and urgent, not a one-off/noisy report.
        description: free-text description of the situation. Run through a
                     trained text classifier (text_urgency_model.py) to catch
                     urgency signal the structured fields alone would miss.

    Returns:
        Decision with priority_score, priority_label, recommended_action, and
        the ML text classifier's contribution (class/confidence/top terms).
    """
    weight = TYPE_WEIGHT.get(report_type, 1.0)
    base_score = severity * weight

    # Small bonus for corroborated reports, capped so one type can't dominate forever.
    duplicate_bonus = min(duplicate_count, 5) * 0.5

    # ML component: learned text urgency signal from the description.
    urgency = predict_urgency(description)

    score = round(base_score + duplicate_bonus + urgency.boost, 2)

    if score >= 12:
        label = "CRITICAL"
    elif score >= 8:
        label = "HIGH"
    elif score >= 4:
        label = "MEDIUM"
    else:
        label = "LOW"

    action = RECOMMENDED_ACTION.get(report_type, "Review manually")

    return Decision(
        priority_score=score,
        priority_label=label,
        recommended_action=action,
        text_urgency_class=urgency.predicted_class,
        text_urgency_confidence=urgency.confidence,
        text_urgency_terms=urgency.top_terms,
    )


# Already using two AI techniques (rule-based scoring + a trained text
# classifier for the ML component; A* search in routing.py is the third,
# for the routing stage). Further extensions if you want more depth:
# - Replace the rule-based structured scorer with a model trained on
#   (type, severity, duplicate_count) -> historical outcome data, once you
#   have real logged incidents to train on.
# - Grow TRAINING_DATA in text_urgency_model.py with more/real examples and
#   retrain (delete urgency_model.joblib, restart the app).
