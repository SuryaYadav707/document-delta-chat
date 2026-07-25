"""Provider-agnostic LLM + embedding client (LangChain). Provider from config
(openai default, bedrock drop-in), credentials from env, temperature=0."""
from __future__ import annotations

from src.config import get_settings


def _llm_cfg(cfg):
    return cfg or get_settings().llm


def build_chat_model(cfg=None, *, model: str | None = None):
    c = _llm_cfg(cfg)
    if c.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or c.chat_model, temperature=c.temperature)
    if c.provider == "bedrock":
        from langchain_aws import ChatBedrock  # drop-in; requires langchain-aws + AWS creds
        return ChatBedrock(model_id=model or getattr(c, "chat_model", None),
                           model_kwargs={"temperature": c.temperature})
    raise ValueError(f"unsupported LLM provider: {c.provider}")


def build_vision_model(cfg=None):
    """Multimodal model for scanned-OCR verify (uses reason_model)."""
    c = _llm_cfg(cfg)
    return build_chat_model(c, model=c.reason_model)


def build_embeddings(cfg=None):
    c = _llm_cfg(cfg)
    if c.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=c.embed_model)
    if c.provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=c.embed_model)
    raise ValueError(f"unsupported LLM provider: {c.provider}")
