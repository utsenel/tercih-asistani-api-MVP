import redis
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True,
                socket_connect_timeout=1,  # Hızlı timeout
                socket_timeout=1
            )
            # Connection test
            self.redis_client.ping()
            logger.info("✅ Redis memory bağlantısı başarılı")
        except:
            logger.warning("❌ Redis yok, memory devre dışı")
            self.redis_client = None
    
    def add_message(self, session_id: str, role: str, content: str):
        if not self.redis_client:
            return
        
        try:
            key = f"chat:{session_id}"
            message = json.dumps({"role": role, "content": content[:500]})  # 500 char limit
            
            self.redis_client.lpush(key, message)
            self.redis_client.ltrim(key, 0, 9)  # Son 10 mesaj
            self.redis_client.expire(key, 3600)  # 1 saat TTL
        except:
            pass  # Silent fail, memory olmadan da çalışsın
    
    def get_history(self, session_id: str, limit: int = 4) -> str:
        if not self.redis_client:
            return ""
        
        try:
            key = f"chat:{session_id}"
            messages = self.redis_client.lrange(key, 0, limit-1)
            
            if not messages:
                return ""
            
            # Compact format
            history = []
            for msg in reversed(messages):  # Eski → yeni sıra
                data = json.loads(msg)
                history.append(f"{data['role']}: {data['content']}")
            
            return "\n".join(history)
        except:
            return ""
