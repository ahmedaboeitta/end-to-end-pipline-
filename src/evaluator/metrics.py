from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import Counter


def calculate_rouge(predictions: list[str], references: list[str]) -> dict:
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)
    
    return {k: sum(v) / len(v) for k, v in scores.items()}


# def calculate_bleu(predictions: list[str], references: list[str]) -> float:
#     """Calculate BLEU score."""
#     bleu = BLEU()
#     # sacrebleu expects list of references per prediction
#     refs = [[ref] for ref in references]
#     result = bleu.corpus_score(predictions, list(zip(*refs)))
#     return result.score / 100  # Normalize to 0-1


def calculate_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Calculate BERTScore (precision, recall, F1)."""
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def calculate_token_f1(predictions: list[str], references: list[str]) -> float:
    """Calculate token-level F1 score."""
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        common = sum((pred_counter & ref_counter).values())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return sum(f1_scores) / len(f1_scores)


# def calculate_exact_match(predictions: list[str], references: list[str]) -> float:
#     """Calculate exact match score."""
#     matches = sum(
#         pred.strip().lower() == ref.strip().lower()
#         for pred, ref in zip(predictions, references)
#     )
#     return matches / len(predictions)


def calculate_all_metrics(predictions: list[str], references: list[str]) -> dict:
    """Calculate all metrics."""
    metrics = {}
    
    # ROUGE
    rouge = calculate_rouge(predictions, references)
    metrics.update(rouge)
    
    # BERTScore
    bertscore = calculate_bertscore(predictions, references)
    metrics.update(bertscore)
    
    # Token F1
    metrics["token_f1"] = calculate_token_f1(predictions, references)
    
    return metrics


def calculate_combined_score(metrics: dict, weights: dict) -> float:
    """Calculate weighted combined score."""
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            score += weight * metrics[metric]
    return score