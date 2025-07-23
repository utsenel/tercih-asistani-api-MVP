from fastapi import FastAPI, HTTPException, Request
import hashlib 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import time

# Local imports - GuidanceTemplates kaldırıldı
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

# Request/Response modelleri
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
    metadata: dict = {}
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Bilgisayar mühendisliği teknik ve analitik...",
                "status": "success", 
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
        "version": AppSettings.API_VERSION,
        "features": {
            "unified_tone": True,  # Yeni özellik
            "csv_analysis": True,
            "vector_search": True,
            "memory_system": True
        }
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "version": AppSettings.API_VERSION,
        "service": AppSettings.API_TITLE,
        "approach": "unified_socratic_tone"  # Güncellenmiş
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest): 
    start_time = time.time()
    
    try:
        # SESSION ID YÖNETİMİ - Sadeleştirilmiş
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        real_ip = request.headers.get("X-Real-IP") 
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")
        
        actual_ip = cf_connecting_ip or real_ip or forwarded_for or client_ip
        if forwarded_for and "," in forwarded_for:
            actual_ip = forwarded_for.split(",")[0].strip()
        
        original_session_id = chat_request.session_id
        
        if not chat_request.session_id or chat_request.session_id in ["ng", "default", ""]:
            user_agent = request.headers.get("user-agent", "unknown")[:50]
            hash_input = f"{actual_ip}_{user_agent}"
            ip_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            chat_request.session_id = f"stable_{ip_hash}"
        
        # SADECE ÖNEMLİ LOGLARI TUTALIM
        if AppSettings.LOG_SESSION_DETAILS:
            logger.debug(f"Session: {original_session_id} → {chat_request.session_id}")
            logger.debug(f"Client IP: {actual_ip}")
        
        logger.info(f"Chat request: session={chat_request.session_id}, message_len={len(chat_request.message)}")
        
        # Chat processor ile işle
        result = await processor.process_message(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        # Processing time hesapla
        processing_time = round(time.time() - start_time, 2)
        
        # SADECE GEREKLİ PERFORMANCE LOG
        if AppSettings.PERFORMANCE_LOGGING:
            logger.info(f"Request completed in {processing_time}s")
        
        return ChatResponse(
            response=result["response"],
            status="success",
            metadata={
                "processing_time": processing_time,
                "session_id": chat_request.session_id,
                "message_length": len(chat_request.message),
                **({
                    "client_ip": actual_ip,
                    "session_transition": f"{original_session_id} → {chat_request.session_id}",
                    **result.get("metadata", {})
                } if AppSettings.DEBUG_MODE else {})
            }
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"Chat endpoint error: {str(e)}")
        
        return ChatResponse(
            response="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            status="error",
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
            "smart_evaluator_corrector": LLMConfigs.SMART_EVALUATOR_CORRECTOR.model,
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
        "approach_config": {
            "unified_tone": True,
            "socratic_method": True,
            "empathetic_responses": True,
            "guidance_categories_removed": True
        },
        "api_info": {
            "title": AppSettings.API_TITLE,
            "version": AppSettings.API_VERSION,
            "description": AppSettings.API_DESCRIPTION
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {AppSettings.API_TITLE} v{AppSettings.API_VERSION}")
    logger.info(f"Unified Socratic Tone System Enabled")
    
    uvicorn.run(
        "main:app", 
        host=AppSettings.API_HOST,
        port=AppSettings.API_PORT,
        reload=True,
        log_level=AppSettings.LOG_LEVEL.lower()
    )
