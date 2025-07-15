import redis
import json
from typing import List, Dict
import logging
import os  

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
                logger.info(f"🔗 Railway Redis'e bağlanıyor: {redis_url[:20]}...")
            else:
                # Local fallback
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    decode_responses=True
                )
                logger.info("🏠 Local Redis'e bağlanıyor...")
            
            # Connection test
            self.redis_client.ping()
            logger.info("✅ Redis memory bağlantısı başarılı")
        except Exception as e:
            logger.warning(f"❌ Redis bağlantı hatası: {e} - Memory devre dışı")
            self.redis_client = None
    
    def add_message(self, session_id: str, role: str, content: str):
        if not self.redis_client:
            logger.warning(f"🚫 Redis client yok - mesaj kaydedilemedi")
            return
        
        try:
            key = f"chat:{session_id}"
            message = json.dumps({"role": role, "content": content[:500]})  # 500 char limit
            
            logger.info(f"💾 MEMORY SAVE - Session: '{session_id}' -> Key: '{key}'")
            logger.info(f"📝 Kaydedilen: {role} -> '{content[:50]}...'")
            
            self.redis_client.lpush(key, message)
            self.redis_client.ltrim(key, 0, 9)  # Son 10 mesaj
            self.redis_client.expire(key, 3600)  # 1 saat TTL
            
            # Verify: Kayıt başarılı mı?
            list_length = self.redis_client.llen(key)
            logger.info(f"✅ Redis key '{key}' toplam mesaj: {list_length}")
            
        except Exception as e:
            logger.error(f"❌ Memory add_message hatası: {e}")
    
    def get_history(self, session_id: str, limit: int = 4) -> str:
        if not self.redis_client:
            logger.warning(f"🚫 Redis client yok - session: {session_id}")
            return ""
        
        try:
            key = f"chat:{session_id}"
            logger.info(f"🔍 MEMORY DEBUG - Session: '{session_id}' -> Redis Key: '{key}'")
            
            # Önce key'in var olup olmadığını kontrol et
            exists = self.redis_client.exists(key)
            logger.info(f"📊 Redis key exists: {exists}")
            
            if not exists:
                logger.info(f"❌ Key '{key}' Redis'te bulunamadı")
                return ""
            
            messages = self.redis_client.lrange(key, 0, limit-1)
            logger.info(f"📝 Bulunan mesaj sayısı: {len(messages)}")
            
            if not messages:
                logger.info(f"📭 Key var ama mesaj yok: {key}")
                return ""
            
            # DEBUG: İlk mesajı göster
            if messages:
                first_msg = json.loads(messages[0])
                logger.info(f"🔍 İlk mesaj: {first_msg['role']} -> '{first_msg['content'][:50]}...'")
            
            # Compact format
            history = []
            for msg in reversed(messages):  # Eski → yeni sıra
                data = json.loads(msg)
                history.append(f"{data['role']}: {data['content']}")
            
            result = "\n".join(history)
            logger.info(f"✅ Döndürülen history: {len(result)} karakter")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Memory get_history hatası: {e}")
            return ""
