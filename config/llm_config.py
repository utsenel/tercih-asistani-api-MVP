import os
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"  # YENİ!

@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float
    max_tokens: int = 500
    max_retries: int = 3
    timeout: int = 100
    
    def to_langchain_params(self) -> Dict[str, Any]:
        if self.provider == LLMProvider.OPENAI:
            return {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "timeout": self.timeout
            }
        elif self.provider == LLMProvider.GOOGLE:
            return {
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "model": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "timeout": self.timeout
            }
        elif self.provider == LLMProvider.ANTHROPIC:  # YENİ!
            return {
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "timeout": self.timeout
            }

class LLMConfigs:
    EVALUATION = LLMConfig(LLMProvider.GOOGLE, "gemini-1.5-flash", 0.3, 50, timeout=30)
    CORRECTION = LLMConfig(LLMProvider.GOOGLE, "gemini-1.5-flash", 0.1, 150, timeout=30)
    SEARCH_OPTIMIZER = LLMConfig(LLMProvider.OPENAI, "gpt-4o-mini", 0.3, 150, timeout=60)    
    CSV_AGENT = LLMConfig(LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022", 0.3, 600, timeout=100)
    FINAL_RESPONSE = LLMConfig(LLMProvider.OPENAI, "gpt-4o-mini", 0.3, 500, timeout=120)    

class LLMFactory:
    @staticmethod
    def create_llm(config):
        params = config.to_langchain_params()
        
        if config.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**params)
        elif config.provider == LLMProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(**params)
        elif config.provider == LLMProvider.ANTHROPIC:  # YENİ!
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(**params)

# Diğer config'ler aynı
class VectorConfig:
    SIMILARITY_TOP_K = 3
    SEARCH_TYPE = "similarity"
    SCORE_THRESHOLD = 0.7

class CSVConfig:
    SAMPLE_ROWS = 68
    MAX_ROWS_FOR_FULL_ANALYSIS = 100

class PerformanceConfig:
    ENABLE_METRICS = True
    LOG_RESPONSE_TIMES = True
    LOG_TOKEN_USAGE = True
    LOG_PROVIDER_DISTRIBUTION = True
    WARNING_THRESHOLD = 5.0
    ERROR_THRESHOLD = 15.0
    TRACK_DAILY_USAGE = True
    DAILY_TOKEN_LIMIT = 1000000

class FallbackConfig:
    ENABLE_FALLBACK = True
    FALLBACK_MAPPINGS = {
        "gemini-1.5-flash": "gpt-4o-mini",
        "gemini-1.5-pro": "gpt-4o-mini",
        "claude-3-5-sonnet-20241022": "gpt-4o",  # YENİ!
        "claude-3-haiku-20240307": "gpt-4o-mini"  # YENİ!
    }
    MAX_FALLBACK_ATTEMPTS = 2
    FALLBACK_DELAY = 1.0
