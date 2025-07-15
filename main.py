from fastapi import FastAPI, HTTPException, Request  # Request ekle
import hashlib 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging

# Local imports
from chat_processor import TercihAsistaniProcessor
from config import AppSettings, ValidationSettings

# Environment değişkenlerini yükle
load_dotenv()

# Logging ayarla - Config'ten
logging.basicConfig(
    level=getattr(logging, AppSettings.LOG_LEVEL),
    format=AppSettings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# FastAPI uygulaması - Config'lerle
app = FastAPI(
    title=AppSettings.API_TITLE,
    description=AppSettings.API_DESCRIPTION,
    version=AppSettings.API_VERSION
)

# CORS ayarları - Config'lerle
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppSettings.CORS_ORIGINS,
    allow_credentials=AppSettings.CORS_CREDENTIALS,
    allow_methods=AppSettings.CORS_METHODS,
    allow_headers=AppSettings.CORS_HEADERS,
)

# Request/Response modelleri - Validation'la
class ChatRequest(BaseModel):
    message: str
    session_id: str = ValidationSettings.DEFAULT_SESSION_ID
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Bilgisayar mühendisliği nasıl?",
                "session_id": "user123"
            }
        }
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validation
        if len(self.message) < ValidationSettings.MIN_MESSAGE_LENGTH:
            raise ValueError("Mesaj çok kısa")
        if len(self.message) > ValidationSettings.MAX_MESSAGE_LENGTH:
            raise ValueError("Mesaj çok uzun")
        if len(self.session_id) > ValidationSettings.MAX_SESSION_ID_LENGTH:
            raise ValueError("Session ID çok uzun")

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    sources: list = []
    metadata: dict = {}
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Bilgisayar mühendisliği teknik ve analitik...",
                "status": "success", 
                "sources": ["YÖK Raporu 2024"],
                "metadata": {"processing_time": 3.2}
            }
        }

# Processor'ı başlat
processor = TercihAsistaniProcessor()

@app.on_event("startup")
async def startup_event():
    """API başlatılırken çalışır"""
    logger.info(f"{AppSettings.API_TITLE} başlatılıyor...")
    await processor.initialize()
    logger.info("API hazır!")

@app.get("/")
async def root():
    """API durumu kontrolü"""
    return {
        "message": f"{AppSettings.API_TITLE} çalışıyor!",
        "status": "active",
        "version": AppSettings.API_VERSION
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "version": AppSettings.API_VERSION,
        "service": AppSettings.API_TITLE
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest): 
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📥 CHAT REQUEST DEBUG:")
        logger.info(f"   📝 Gelen mesaj: '{chat_request.message[:50]}...'")
        logger.info(f"   🔑 Frontend session_id: '{chat_request.session_id}'")
        
        # CLIENT IP DETAYLI ANALİZİ
        client_ip = request.client.host
        
        # Proxy header'larını kontrol et
        forwarded_for = request.headers.get("X-Forwarded-For")
        real_ip = request.headers.get("X-Real-IP") 
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")  # Cloudflare
        
        logger.info(f"🌐 IP ANALİZİ:")
        logger.info(f"   🔌 request.client.host: '{client_ip}'")
        logger.info(f"   📡 X-Forwarded-For: '{forwarded_for}'")
        logger.info(f"   🌍 X-Real-IP: '{real_ip}'")
        logger.info(f"   ☁️ CF-Connecting-IP: '{cf_connecting_ip}'")
        
        # En güvenilir IP'yi seç
        actual_ip = cf_connecting_ip or real_ip or forwarded_for or client_ip
        if forwarded_for and "," in forwarded_for:
            actual_ip = forwarded_for.split(",")[0].strip()
        
        logger.info(f"   ✅ Selected IP: '{actual_ip}'")
        
        # SESSION ID GENERATION WITH STABLE IP
        original_session_id = chat_request.session_id
        
        if not chat_request.session_id or chat_request.session_id in ["ng", "default", ""]:
            # STABLE HASH - IP + User-Agent kombine et
            user_agent = request.headers.get("user-agent", "unknown")[:50]  # İlk 50 karakter
            
            # Hash input'u stabil olsun
            hash_input = f"{actual_ip}_{user_agent}"
            ip_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            chat_request.session_id = f"stable_{ip_hash}"
            
            logger.info(f"🔄 SESSION ID GENERATION:")
            logger.info(f"   📊 Hash input: '{hash_input[:80]}...'")
            logger.info(f"   🎯 Generated session: '{chat_request.session_id}'")
        else:
            logger.info(f"✅ Frontend session kullanılıyor: '{chat_request.session_id}'")
        
        logger.info(f"📊 SESSION TRANSITION:")
        logger.info(f"   📥 Original: '{original_session_id}'")
        logger.info(f"   📤 Final: '{chat_request.session_id}'")
        
        # Redis'te bu session'ın durumunu kontrol et
        try:
            # Memory instance'a erişim
            if hasattr(processor, 'memory') and processor.memory and processor.memory.redis_client:
                redis_key = f"chat:{chat_request.session_id}"
                exists = processor.memory.redis_client.exists(redis_key)
                
                if exists:
                    message_count = processor.memory.redis_client.llen(redis_key)
                    logger.info(f"📊 EXISTING SESSION: '{redis_key}' - {message_count} mesaj")
                else:
                    logger.info(f"🆕 NEW SESSION: '{redis_key}' - ilk mesaj")
        except Exception as redis_check_error:
            logger.warning(f"⚠️ Redis session check hatası: {redis_check_error}")
        
        # Chat processor ile işle
        result = await processor.process_message(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        # Processing time hesapla
        processing_time = round(time.time() - start_time, 2)
        
        return ChatResponse(
            response=result["response"],
            status="success",
            sources=result.get("sources", []),
            metadata={
                "processing_time": processing_time,
                "session_id": chat_request.session_id,
                "message_length": len(chat_request.message),
                "client_ip": actual_ip,  # Debug için
                "session_transition": f"{original_session_id} → {chat_request.session_id}"
            }
        )
        
    except ValueError as e:
        # Validation hatası
        logger.warning(f"Validation hatası: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Genel hata
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"Chat endpoint hatası: {str(e)}")
        
        return ChatResponse(
            response="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            status="error",
            sources=[],
            metadata={
                "processing_time": processing_time,
                "error": str(e)[:100]
            }
        )

@app.get("/test-connection")
async def test_connections():
    """
    Bağlantıları test et
    """
    try:
        test_results = await processor.test_all_connections()
        return {
            "status": "success", 
            "results": test_results,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/debug-astra")
async def debug_astra():
    """
    AstraDB doküman yapısını debug et
    """
    try:
        debug_results = await processor.debug_astra_documents()
        return {
            "status": "success",
            "debug_info": debug_results
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug-csv")
async def debug_csv():
    """
    CSV debug endpoint
    """
    try:
        debug_results = await processor.debug_csv_data()
        return {
            "status": "success",
            "debug_info": debug_results
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/config-info")
async def config_info():
    """
    Config bilgilerini göster (sensitive bilgiler hariç)
    """
    from config import LLMConfigs, VectorConfig, CSVConfig
    
    return {
        "llm_models": {
            "evaluation": LLMConfigs.EVALUATION.model,
            "correction": LLMConfigs.CORRECTION.model,
            "search_optimizer": LLMConfigs.SEARCH_OPTIMIZER.model,
            "csv_agent": LLMConfigs.CSV_AGENT.model,
            "final_response": LLMConfigs.FINAL_RESPONSE.model
        },
        "vector_config": {
            "top_k": VectorConfig.SIMILARITY_TOP_K,
            "search_type": VectorConfig.SEARCH_TYPE,
            "score_threshold": VectorConfig.SCORE_THRESHOLD
        },
        "csv_config": {
            "sample_rows": CSVConfig.SAMPLE_ROWS,
            "max_rows_full_analysis": CSVConfig.MAX_ROWS_FOR_FULL_ANALYSIS
        },
        "api_info": {
            "title": AppSettings.API_TITLE,
            "version": AppSettings.API_VERSION,
            "description": AppSettings.API_DESCRIPTION
        }
    }

if __name__ == "__main__":
    import uvicorn
    import time
    
    logger.info(f"Starting {AppSettings.API_TITLE} v{AppSettings.API_VERSION}")
    
    uvicorn.run(
        "main:app", 
        host=AppSettings.API_HOST,
        port=AppSettings.API_PORT,
        reload=True,
        log_level=AppSettings.LOG_LEVEL.lower()
    )
