import redis
import json
from typing import List, Dict
import logging
import os  
from config import AppSettings

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self):
        try:
            # Railway Redis URL'i kullan
            redis_url = os.getenv('REDIS_URL') or os.getenv('REDIS_PRIVATE_URL')
            
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                if AppSettings.LOG_MEMORY_OPERATIONS:
                    logger.debug(f"Railway Redis'e bağlanıyor: {redis_url[:20]}...")
            else:
                # Local fallback
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    decode_responses=True
                )
                if AppSettings.LOG_MEMORY_OPERATIONS:
                    logger.debug("Local Redis'e bağlanıyor...")
            
            # Connection test
            self.redis_client.ping()
            logger.info("Redis memory connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - Memory disabled")
            self.redis_client = None
    
    def add_message(self, session_id: str, role: str, content: str):
        if not self.redis_client:
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug("Redis client unavailable - message not saved")
            return
        
        try:
            key = f"chat:{session_id}"
            message = json.dumps({"role": role, "content": content[:500]})  # 500 char limit
            
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"Memory save: {role} -> {session_id}")
            
            self.redis_client.lpush(key, message)
            self.redis_client.ltrim(key, 0, 9)  # Son 10 mesaj
            self.redis_client.expire(key, 3600)  # 1 saat TTL
            
        except Exception as e:
            logger.error(f"Memory add_message error: {e}")
    
    def get_history(self, session_id: str, limit: int = 4) -> str:
        if not self.redis_client:
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"Redis client unavailable - session: {session_id}")
            return ""
        
        try:
            key = f"chat:{session_id}"
            
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"Memory fetch: {session_id}")
            
            # Key kontrolü
            exists = self.redis_client.exists(key)
            if not exists:
                if AppSettings.LOG_MEMORY_OPERATIONS:
                    logger.debug(f"No history found for session: {session_id}")
                return ""
            
            messages = self.redis_client.lrange(key, 0, limit-1)
            
            if not messages:
                if AppSettings.LOG_MEMORY_OPERATIONS:
                    logger.debug(f"Empty history for session: {session_id}")
                return ""
            
            # Compact format
            history = []
            for msg in reversed(messages):  # Eski → yeni sıra
                data = json.loads(msg)
                history.append(f"{data['role']}: {data['content']}")
            
            result = "\n".join(history)
            
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"History retrieved: {len(messages)} messages, {len(result)} chars")
            
            return result
            
        except Exception as e:
            logger.error(f"Memory get_history error: {e}")
            return ""
    def log_analytics(self, data: dict):
        """Analytics verilerini Redis'e kaydet"""
        if not self.redis_client:
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug("Redis client unavailable - analytics not saved")
            return
        
        try:
            # Timestamp ekleme eğer yoksa
            if "ts" not in data:
                from datetime import datetime
                data["ts"] = datetime.now().isoformat()
            
            # JSON string'e çevir
            log_entry = json.dumps(data, ensure_ascii=False)
            
            # Redis LIST'e ekle
            self.redis_client.lpush("analytics:logs", log_entry)
            
            # İsteğe bağlı: son 1000 kaydı tut (disk alanı için)
            self.redis_client.ltrim("analytics:logs", 0, 999)
            
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"Analytics logged: session={data.get('session', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Analytics logging error: {e}")

    def get_all_logs(self, limit: int = 100) -> List[dict]:
        """Analytics loglarını al"""
        if not self.redis_client:
            return []
        
        try:
            raw_logs = self.redis_client.lrange("analytics:logs", 0, limit-1)
            logs = []
            
            for raw_log in raw_logs:
                try:
                    log_data = json.loads(raw_log)
                    logs.append(log_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Log parse error: {e}")
                    continue
            
            return logs
            
        except Exception as e:
            logger.error(f"Get analytics logs error: {e}")
            return []
