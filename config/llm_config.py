import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"

@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float
    max_tokens: int = 400
    max_retries: int = 3
    timeout: int = 100
    fallback_provider: Optional[LLMProvider] = None
    fallback_model: Optional[str] = None
    
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
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
                "model": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "max_retries": self.max_retries,
                "timeout": self.timeout
            }
        elif self.provider == LLMProvider.ANTHROPIC:
            return {
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
    
    def get_fallback_config(self):
        """Fallback config döndür"""
        if self.fallback_provider and self.fallback_model:
            return LLMConfig(
                provider=self.fallback_provider,
                model=self.fallback_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
                timeout=self.timeout
            )
        return None

class LLMConfigs:
    """
    Güncellenmiş LLM konfigürasyonları - Smart Evaluator-Corrector ile
    """
    
    # YENİ: Birleştirilmiş Smart Evaluator-Corrector
    SMART_EVALUATOR_CORRECTOR = LLMConfig(
        provider=LLMProvider.OPENAI, 
        model="gpt-4o-mini",  
        temperature=0.2,  # Düşük temperature - consistent output için
        max_tokens=150,   # Artırıldı - context analysis için
        timeout=45,
        max_retries=2,
        fallback_provider=LLMProvider.GOOGLE,
        fallback_model="gemini-1.5-pro"
    )
    
    # KALAN MODELLER - değişmedi
    CSV_AGENT = LLMConfig(
        provider=LLMProvider.OPENAI, 
        model="gpt-4o-mini", 
        temperature=0.2, 
        max_tokens=250,  
        timeout=60,
        max_retries=2, 
        fallback_provider=LLMProvider.ANTHROPIC,
        fallback_model="claude-3-5-sonnet-20241022"
    )
    
    FINAL_RESPONSE = LLMConfig(
        provider=LLMProvider.OPENAI, 
        model="gpt-4o", 
        temperature=0.3, 
        max_tokens=250, 
        timeout=60,
        max_retries=2, 
        fallback_provider=LLMProvider.ANTHROPIC,
        fallback_model="claude-3-5-sonnet-20241022"
    )

class LLMFactory:
    """Gelişmiş LLM Factory - Fallback desteği"""
    
    @staticmethod
    def create_llm(config: LLMConfig, use_fallback: bool = False):
        """
        LLM oluştur, hata durumunda fallback kullan
        """
        target_config = config
        
        if use_fallback and config.get_fallback_config():
            target_config = config.get_fallback_config()
            logger.info(f"🔄 Fallback kullanılıyor: {target_config.provider.value} - {target_config.model}")
        
        try:
            params = target_config.to_langchain_params()
            
            if target_config.provider == LLMProvider.OPENAI:
                api_key = params.get("api_key")
                from langchain_openai import ChatOpenAI
                if not api_key:
                    raise ValueError("OPENAI_API_KEY bulunamadı")
                return ChatOpenAI(**params)
                
            elif target_config.provider == LLMProvider.GOOGLE:
                api_key = params.get("google_api_key")
                from langchain_google_genai import ChatGoogleGenerativeAI
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY bulunamadı")
                return ChatGoogleGenerativeAI(**params)
                
            elif target_config.provider == LLMProvider.ANTHROPIC:
                api_key = params.get("anthropic_api_key")
                from langchain_anthropic import ChatAnthropic
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY bulunamadı")
                return ChatAnthropic(**params)
            
            else:
                raise ValueError(f"Desteklenmeyen provider: {target_config.provider}")
                
        except Exception as e:
            logger.error(f"❌ LLM oluşturma hatası ({target_config.provider.value}): {e}")
            raise

    @staticmethod
    def create_llm_with_fallback(config: LLMConfig):
        """
        Primary'yi dene, başarısız olursa fallback'i kullan
        """
        try:
            return LLMFactory.create_llm(config, use_fallback=False)
        except Exception as primary_error:
            logger.warning(f"⚠️ Primary LLM hatası: {primary_error}")
            
            if config.get_fallback_config():
                try:
                    logger.info(f"🔄 Fallback deneniyor...")
                    return LLMFactory.create_llm(config, use_fallback=True)
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback de başarısız: {fallback_error}")
                    raise fallback_error
            else:
                raise primary_error

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
        "claude-3-5-sonnet-20241022": "gpt-4o",
        "claude-3-haiku-20240307": "gpt-4o-mini",
        "gpt-4o-mini":"gemini-1.5-pro",
        "gpt-4o":"claude-3-5-sonnet-20241022"
    }
    MAX_FALLBACK_ATTEMPTS = 2
    FALLBACK_DELAY = 1.0
