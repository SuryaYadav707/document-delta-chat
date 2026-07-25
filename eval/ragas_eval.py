"""Ragas metrics for the chat pipeline.

Runs faithfulness, response relevancy, context precision/recall, and factual
correctness. The judge LLM is `eval.judge_model` (gpt-4o) — DELIBERATELY different
from the answer model (gpt-4o-mini) to avoid self-preference bias. Best-effort:
returns {"skipped": ...} if ragas isn't installed.
"""
from __future__ import annotations

import sys
import types

# ragas 0.2.x imports a langchain module removed in 0.3; stub it (unused — we pass
# our own LLM/embeddings to evaluate()).
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = object
sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)


def run_ragas(samples: list[dict], cfg) -> dict:
    """samples: [{question, contexts:[str], answer, gold}]. Returns metric -> mean."""
    if not samples:
        return {"skipped": "no samples"}
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (FactualCorrectness, Faithfulness,
                                   LLMContextPrecisionWithReference, LLMContextRecall,
                                   ResponseRelevancy)

        judge = LangchainLLMWrapper(ChatOpenAI(model=cfg.eval.judge_model, temperature=0))
        embed = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=cfg.llm.embed_model))
        dataset = EvaluationDataset(samples=[
            SingleTurnSample(user_input=s["question"], response=s["answer"],
                             retrieved_contexts=s["contexts"], reference=s["gold"])
            for s in samples])
        metrics = [Faithfulness(), ResponseRelevancy(),
                   LLMContextPrecisionWithReference(), LLMContextRecall(),
                   FactualCorrectness()]
        result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embed,
                          show_progress=False)
        df = result.to_pandas()
        return {col: round(float(df[col].mean()), 3)
                for col in df.select_dtypes("number").columns}
    except Exception as e:
        return {"error": str(e)[:150]}
