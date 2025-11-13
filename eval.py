from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_summary(gold: str, generated: str):
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(gold, generated)

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        [gold.split()],
        generated.split(),
        smoothing_function=smoothie
    )

    return {
        "rouge1": rouge['rouge1'].fmeasure,
        "rougeL": rouge['rougeL'].fmeasure,
        "bleu": bleu
    }
