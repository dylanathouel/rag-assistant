"""
eval/run_ragas.py — Évaluation RAGAS du pipeline RAG (ragas >= 0.2.0).

4 métriques :
  context_precision  : les chunks récupérés sont-ils pertinents pour la question ?
  context_recall     : le contexte couvre-t-il la bonne réponse ?
  faithfulness       : la réponse est-elle ancrée dans le contexte (sans hallucination) ?
  answer_relevancy   : la réponse répond-elle à la question ?

Installation préalable :
    docker compose exec rag-app pip install -r requirements-eval.txt
    # ou en local :
    pip install -r requirements-eval.txt

Usage :
    python eval/run_ragas.py                        # hybrid search (défaut)
    python eval/run_ragas.py --no-hybrid            # vectoriel pur (baseline S1)
    python eval/run_ragas.py --output results.json  # sauvegarder les scores
    python eval/run_ragas.py --limit 5              # tester sur 5 questions seulement
"""
import json
import argparse
import sys
import os
import time
from types import ModuleType

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ragas 0.2.x importe ChatVertexAI depuis un sous-module supprimé dans
# langchain-community 0.3.x. Ce shim crée un module factice pour débloquer l'import.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _shim = ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from pathlib import Path

DATASET_PATH = Path(__file__).parent / "dataset.json"


def build_evaluator_llm():
    """LLM pour ragas — Claude Haiku 4.5 via OpenRouter."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from config import OPENROUTER_API_KEY

    return LangchainLLMWrapper(ChatOpenAI(
        model="anthropic/claude-haiku-4.5",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=2048,
        n=1,
    ))


def run_evaluation(hybrid: bool = True, limit: int | None = None) -> dict:
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
    )
    from query import search_chunks, generate_answer

    raw = json.loads(DATASET_PATH.read_text())
    if limit:
        raw = raw[:limit]

    samples = []
    latencies = []

    for i, item in enumerate(raw, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i:2d}/{len(raw)}] {question[:70]}...")
        t0 = time.time()

        chunks = search_chunks(question, hybrid=hybrid)
        answer = generate_answer(question, chunks)
        latencies.append(time.time() - t0)

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=[c[0] for c in chunks],
            response=answer,
            reference=ground_truth,
        ))

    eval_dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            # ResponseRelevancy retiré : nécessite des embeddings incompatibles avec notre setup
        ],
        llm=build_evaluator_llm(),
    )

    df = result.to_pandas()
    avg = df.mean(numeric_only=True).to_dict()

    return {
        "context_precision": round(avg.get("llm_context_precision_with_reference", 0), 4),
        "context_recall": round(avg.get("llm_context_recall", 0), 4),
        "faithfulness": round(avg.get("faithfulness", 0), 4),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "n_questions": len(samples),
        "hybrid": hybrid,
        "_all_scores": avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    parser.add_argument("--output", default=None, help="Fichier JSON pour sauvegarder les scores")
    parser.add_argument("--limit", type=int, default=None, help="Limiter à N questions")
    args = parser.parse_args()

    mode = "hybrid" if args.hybrid else "vector-only"
    print(f"\n🧪 RAGAS evaluation — mode : {mode}")
    if args.limit:
        print(f"   Limit : {args.limit} questions")
    print()

    scores = run_evaluation(hybrid=args.hybrid, limit=args.limit)

    print(f"\n{'='*45}")
    print(f"  RAGAS Results ({mode})")
    print(f"{'='*45}")
    print(f"  context_precision  : {scores['context_precision']:.4f}")
    print(f"  context_recall     : {scores['context_recall']:.4f}")
    print(f"  faithfulness       : {scores['faithfulness']:.4f}")
    print(f"  ---")
    print(f"  avg latency        : {scores['avg_latency_s']:.2f}s")
    print(f"  questions tested   : {scores['n_questions']}")
    print(f"{'='*45}\n")

    if args.output:
        out = {k: v for k, v in scores.items() if k != "_all_scores"}
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"💾 Scores sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
