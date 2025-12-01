import spacy
from collections import Counter

"""
Analysis of extracted facts linguistic composition (figure 3 in the paper).
"""

nlp = spacy.load("en_core_web_sm")

def classify_span(span_text):
    doc = nlp(span_text)

    has_finite_verb = any(token.tag_ in {"VBD", "VBP", "VBZ", "MD"} for token in doc)
    num_clauses = sum(1 for token in doc if token.dep_ == "ROOT" and token.pos_ in {'VERB', 'AUX'})
    has_discourse = any(token.dep_ == "mark" for token in doc)
    has_multiple_clauses = num_clauses > 1

    if not has_finite_verb:
        return "Phrase"
    elif not has_multiple_clauses and not has_discourse:
        return "Simple Clause"
    else:
        return "Complex Sentence"

def linguistic_analysis(spans):
    classified_spans = [classify_span(span) for span in spans]

    span_stats = Counter(classified_spans)
    total_spans = len(spans)

    print("Span Linguistic Composition:")
    for span_type, count in span_stats.items():
        percentage = (count / total_spans) * 100
        print(f"{span_type}: {count} spans ({percentage:.2f}%)")
        
    return classified_spans

if __name__ == '__main__':
    spans = [
        "the large house",  # phrase
        "She went home",  # simple clause
        "She went home because she was tired.",  # complex sentence
    ]

    linguistic_analysis(spans)