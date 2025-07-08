"""
Genel uygulama ayarları
"""
import os
from typing import List

class AppSettings:
    """Uygulama genel ayarları"""
    
    # API Ayarları
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_TITLE = "Tercih Asistanı API"
    API_DESCRIPTION = "Üniversite tercih rehberliği için AI asistan"
    API_VERSION = "1.0.0"
    
    # CORS Ayarları
    CORS_ORIGINS = ["*"]  # Production'da değiştirin
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Logging Ayarları
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class DatabaseSettings:
    """Veritabanı ayarları"""
    
    # AstraDB
    ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION")
    ASTRA_DB_ENVIRONMENT = os.getenv("ASTRA_DB_ENVIRONMENT")
    
    # CSV
    CSV_FILE_PATH = "./data/cbiko_birlesik_genisletilmis_ai_friendly.csv"
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MessageSettings:
    """Mesaj işleme ayarları"""
    
    # Hata Mesajları
    ERROR_GENERAL = "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    ERROR_EXPERTISE_OUT = "Uzmanlaştığım alanın dışında bir soru olduğundan cevap veremiyorum. Yardımcı olabileceğim başka bir konu var mıydı?"
    
    # Kaynak İsimleri
    SOURCES = {
        "YOK_REPORT": "YÖK Üniversite İzleme ve Değerlendirme Genel Raporu 2024",
        "IZU_GUIDE": "İZÜ YKS Tercih Rehberi", 
        "GENERAL": "Genel rehberlik bilgisi",
        "UNIVERI_DB": "Cumhurbaşkanlığı UNİ-VERİ veritabanı (2024)"
    }
    
    # Response Limits
    MAX_RESPONSE_LENGTH = 1000
    MIN_RESPONSE_LENGTH = 50

class ValidationSettings:
    """Validasyon ayarları"""
    
    # Mesaj Validasyonu
    MIN_MESSAGE_LENGTH = 1
    MAX_MESSAGE_LENGTH = 500
    
    # Session Validasyonu
    MAX_SESSION_ID_LENGTH = 50
    DEFAULT_SESSION_ID = "default"