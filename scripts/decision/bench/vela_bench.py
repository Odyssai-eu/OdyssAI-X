"""Fixed request bank for the Vela 3.0 oracle comparison: router questions (decision / task adapters), a base question,
an opt-in row, and span questions on every route (PII set, open labels with definitions, relation, evidence, halu)."""
import json
from pathlib import Path


def requests(package: Path):
    rq = json.loads((package / "router_questions.json").read_text())["trained_questions"]
    router = {k: v["question"] for k, v in rq.items() if "question" in v}
    states = [
        "Can you double-check whether the Eiffel Tower is taller than the Empire State Building?",
        "Draw me a picture of a cat wearing a spacemen helmet, and explain how you made it.",
        "That answer did not work at all, the script still crashes on line 12.",
        "Translate this paragraph into German and keep the formatting.",
    ]
    base = {"type": "choice", "instructions": "Which language is the request written in?",
            "criteria": {"english": None, "german": None, "french": None}}
    out = []
    for s in states:
        qs = {k: router[k] for k in ("factcheck", "modality", "domain", "feedback") if k in router}
        qs["lang"] = base
        qs["urgent"] = {"type": "noul", "instructions": "Is this request urgent?", "decision_adapter": True}
        out.append((s, qs))
    out.append((
        {"user": "Hi, I'm Maria Gonzalez, born 12 March 1987. Email me at maria.g@example.com or call +1 415 555 0134. "
                 "I live at 221B Baker Street, London NW1 6XE and work for Acme Corp."},
        {"pii": {"type": "span", "criteria": {k: None for k in ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS",
                                                                 "DATE_TIME", "ORGANIZATION")}},
         "ner": {"type": "span", "criteria": {"person": None, "organization": None, "city": None}},
         "ner_given": {"type": "span", "criteria": {"person": "a named human being", "company": "a business or firm"}},
         "route": {"type": "choice", "instructions": "What does the user share?",
                   "criteria": {"contact": "contact details", "complaint": "a complaint", "question": "a question"}}}))
    out.append((
        "Steve Jobs founded Apple in Cupertino together with Steve Wozniak in 1976. Tim Cook became CEO in 2011.",
        {"rel": {"type": "span", "instructions": "Which spans are the founder of Apple?", "criteria": {"founder": None}},
         "rel2": {"type": "span", "instructions": "Which spans are related to Apple by these relations?",
                  "criteria": {"founder": None, "chief executive": None}}}))
    out.append((
        {"question": "What is the capital of Australia?",
         "context": "Canberra is the capital city of Australia. Sydney is its largest city and Melbourne the second largest.",
         "answer": "The capital of Australia is Sydney, which is also its largest city."},
        {"halu": {"type": "span", "criteria": ["unsupported"]},
         "evidence": {"type": "span", "target": "context", "criteria": ["evidence"]}}))
    return out
