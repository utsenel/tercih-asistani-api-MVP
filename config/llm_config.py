"""
LLM Konfigürasyonları - Tüm model ayarları burada
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LLMConfig:
    """LLM model konfigürasyonu"""
    model: str
    temperature: float
    max_tokens: int = 500
    max_retries: int = 3
    timeout: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """OpenAI parametrelerine dönüştür"""
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout": self.timeout
        }

# LLM Model Konfigürasyonları
class LLMConfigs:
    """Tüm LLM konfigürasyonları"""
    
    # Soru uygunluk değerlendirmesi - Daha karmaşık karar verme
    EVALUATION = LLMConfig(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=100,
        max_retries=3,
        timeout=60
    )
    
    # Soru düzeltme - Deterministik olmalı
    CORRECTION = LLMConfig(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=200,
        max_retries=3,
        timeout=60
    )
    
    # Arama sorgusu optimizasyonu - Yaratıcı olabilir
    SEARCH_OPTIMIZER = LLMConfig(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=150,
        max_retries=3,
        timeout=60
    )
    
    # CSV agent - Analitik düşünme
    CSV_AGENT = LLMConfig(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=600,
        max_retries=3,
        timeout=100
    )
    
    # Final yanıt - En önemli, kaliteli olmalı
    FINAL_RESPONSE = LLMConfig(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=500,
        max_retries=3,
        timeout=120
    )

# Vector Search Ayarları
class VectorConfig:
    """Vector arama konfigürasyonları"""
    SIMILARITY_TOP_K = 3
    SEARCH_TYPE = "similarity"
    SCORE_THRESHOLD = 0.7

# CSV Analiz Ayarları  
class CSVConfig:
    """CSV analiz konfigürasyonları"""
    SAMPLE_ROWS = 68  # CSV'den kaç satır örnek gönderilecek
    MAX_ROWS_FOR_FULL_ANALYSIS = 100  # Tam analiz için maksimum satır
