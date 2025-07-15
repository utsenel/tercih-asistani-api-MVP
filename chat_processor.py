import os
import pandas as pd
import re
from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import math
import json
import asyncio
import re
from astrapy import DataAPIClient
from openai import OpenAI
from memory import ConversationMemory
from langchain_anthropic import ChatAnthropic

# Config imports
from config import (
    LLMConfigs, LLMProvider, VectorConfig, CSVConfig,
    PromptTemplates, CSV_KEYWORDS,
    DatabaseSettings, MessageSettings
)

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(config):
        """Gelişmiş hata yönetimi ile LLM oluştur"""
        try:
            params = config.to_langchain_params()
            logger.info(f"🤖 LLM oluşturuluyor: {config.provider.value} - {config.model}")
            
            if config.provider == LLMProvider.OPENAI:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(**params)
            elif config.provider == LLMProvider.GOOGLE:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(**params)
            elif config.provider == LLMProvider.ANTHROPIC:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(**params)
            else:
                raise ValueError(f"Desteklenmeyen provider: {config.provider}")
                
        except Exception as e:
            logger.error(f"❌ LLM oluşturma hatası ({config.provider.value}): {e}")
            raise

class TercihAsistaniProcessor:
    """
    HIZLANDIRILMIŞ VERSION - Railway timeout fix
    """
    
    def __init__(self):
        self.llm_evaluation = None
        self.llm_correction = None
        self.llm_search_optimizer = None
        self.llm_csv_agent = None
        self.llm_final = None
        
        # Native components
        self.openai_client = None
        self.astra_database = None
        self.astra_collection = None
        
        self.csv_data = None
        self.memory = ConversationMemory() 
        
        # Config'lerden prompt'ları al
        self.evaluation_prompt = ChatPromptTemplate.from_template(PromptTemplates.EVALUATION)
        self.correction_prompt = ChatPromptTemplate.from_template(PromptTemplates.CORRECTION)
        self.search_optimizer_prompt = ChatPromptTemplate.from_template(PromptTemplates.SEARCH_OPTIMIZER)
        self.csv_agent_prompt = ChatPromptTemplate.from_template(PromptTemplates.CSV_AGENT)
        self.final_prompt = ChatPromptTemplate.from_template(PromptTemplates.FINAL_RESPONSE)

    def get_embedding(self, text: str) -> List[float]:
        """OpenAI embedding oluştur"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Embedding oluşturma hatası: {e}")
            raise

    async def initialize(self):
        """HIZLANDIRILMIŞ başlatma"""
        try:
            logger.info("🚀 TercihAsistaniProcessor başlatılıyor...")
            
            # API Key kontrolü (basit)
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY zorunlu!")
            
            # OpenAI client'ı başlat
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ OpenAI client başlatıldı")
            
            # HIZLI LLM'leri paralel başlat
            await self._initialize_llms_fast()
            
            # AstraDB bağlantısı
            await self._initialize_astradb_native()
            
            # CSV verilerini yükle
            await self._initialize_csv()
                
            logger.info("✅ TercihAsistaniProcessor başarıyla başlatıldı")
            
        except Exception as e:
            logger.error(f"❌ Initialization hatası: {e}")
            raise

    async def _initialize_llms_fast(self):
        """SADECE OPENAI modelleri - maksimum hız"""
        try:
            # Tüm kritik işlemler için hızlı OpenAI modelleri
            fast_config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=300,
                timeout=15  # 15 saniye timeout
            )
            
            # Tüm LLM'ler aynı hızlı model
            self.llm_evaluation = LLMFactory.create_llm(fast_config)
            self.llm_correction = LLMFactory.create_llm(fast_config)
            self.llm_search_optimizer = LLMFactory.create_llm(fast_config)
            self.llm_csv_agent = LLMFactory.create_llm(fast_config)
            self.llm_final = LLMFactory.create_llm(fast_config)
            
            logger.info("✅ Tüm LLM'ler hızlı OpenAI modeli ile başlatıldı")
            
        except Exception as e:
            logger.error(f"❌ Fast LLM initialization hatası: {e}")
            raise

    async def _initialize_astradb_native(self):
        """Hızlı AstraDB bağlantısı"""
        try:
            logger.info("🔌 AstraDB native API bağlantısı...")
            
            astra_client = DataAPIClient(DatabaseSettings.ASTRA_DB_TOKEN)
            self.astra_database = astra_client.get_database(DatabaseSettings.ASTRA_DB_API_ENDPOINT)
            self.astra_collection = self.astra_database.get_collection(DatabaseSettings.ASTRA_DB_COLLECTION)
            
            logger.info(f"✅ AstraDB başarılı")
            
        except Exception as e:
            logger.error(f"❌ AstraDB hatası: {e}")
            self.astra_collection = None

    async def _initialize_csv(self):
        """Hızlı CSV yükleme"""
        try:
            csv_path = DatabaseSettings.CSV_FILE_PATH
            if csv_path and os.path.exists(csv_path):
                self.csv_data = pd.read_csv(csv_path)
                logger.info(f"✅ CSV: {len(self.csv_data)} satır")
            else:
                self.csv_data = None
        except Exception as e:
            logger.warning(f"CSV hatası: {e}")
            self.csv_data = None

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """SÜPER HIZLI mesaj işleme - max 20 saniye"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"⚡ HIZLI işleme başlıyor - Session: {session_id}")
            
            # Adım 1: Hızlı değerlendirme (2s max)
            evaluation_task = asyncio.create_task(self._evaluate_question_ultra_fast(message))
            
            # Adım 2: Paralel düzeltme ve context alma (başla ama bekle)
            correction_task = asyncio.create_task(self._correct_question_fast(message))
            
            # Evaluation sonucunu bekle
            evaluation_result = await evaluation_task
            
            if evaluation_result == "Uzmanlık dışı soru":
                return {"response": MessageSettings.ERROR_EXPERTISE_OUT, "sources": []}
            
            if evaluation_result == "SELAMLAMA":
                return {
                    "response": "Merhaba! Size üniversite tercih konularında yardımcı olabilirim. Hangi konuda bilgi almak istiyorsunuz?",
                    "sources": []
                }
            
            # Düzeltilmiş soruyu al
            corrected_question = await correction_task
            
            # Adım 3: GERÇEK PARALEL context alma (en kritik optimizasyon)
            context_tasks = [
                asyncio.create_task(self._get_vector_context_ultra_fast(corrected_question)),
                asyncio.create_task(self._get_csv_context_ultra_fast(corrected_question))
            ]
            
            # Paralel context'leri bekle (max 10s timeout)
            try:
                context1, context2 = await asyncio.wait_for(
                    asyncio.gather(*context_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Context timeout, fallback kullanılıyor")
                context1 = "Vector arama zaman aşımı"
                context2 = "CSV analizi zaman aşımı"
            
            # Exception handling
            if isinstance(context1, Exception):
                context1 = "Vector arama başarısız"
            if isinstance(context2, Exception):
                context2 = "CSV analizi başarısız"
            
            # Adım 4: Hızlı final yanıt
            conversation_history = self.memory.get_history(session_id)
            final_response = await self._generate_final_response_ultra_fast(
                question=corrected_question,
                context1=context1,
                context2=context2,
                history=conversation_history
            )

            # Memory'ye kaydet (async olarak, bekleme)
            asyncio.create_task(self._save_to_memory_async(session_id, message, final_response))

            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"⚡ İşlem tamamlandı: {elapsed:.2f}s")

            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2)
            }
            
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Hızlı işleme hatası ({elapsed:.2f}s): {e}")
            return {"response": MessageSettings.ERROR_GENERAL, "sources": []}

    async def _save_to_memory_async(self, session_id: str, user_msg: str, assistant_msg: str):
        """Memory'ye asenkron kaydet - ana işlemi bloklamaz"""
        try:
            self.memory.add_message(session_id, "user", user_msg)
            self.memory.add_message(session_id, "assistant", assistant_msg)
        except:
            pass  # Memory hatası ana işlemi etkilemesin

    async def _evaluate_question_ultra_fast(self, question: str) -> str:
        """Ultra hızlı değerlendirme - 2s max"""
        try:
            if not self.llm_evaluation:
                return "UYGUN"
                
            # Timeout ile koruma
            result = await asyncio.wait_for(
                self.llm_evaluation.ainvoke(self.evaluation_prompt.format(question=question)),
                timeout=3.0
            )
            
            evaluation_result = result.content.strip().upper()
            
            if "SELAMLAMA" in evaluation_result:
                return "SELAMLAMA"
            elif "UYGUN" in evaluation_result:
                return "UYGUN"
            else:
                return "Uzmanlık dışı soru"
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ Evaluation timeout, varsayılan UYGUN")
            return "UYGUN"
        except Exception as e:
            logger.error(f"❌ Evaluation hatası: {e}")
            return "UYGUN"

    async def _correct_question_fast(self, question: str) -> str:
        """Hızlı soru düzeltme"""
        try:
            if not self.llm_correction:
                return question
                
            result = await asyncio.wait_for(
                self.llm_correction.ainvoke(self.correction_prompt.format(question=question)),
                timeout=5.0
            )
            
            return result.content.strip()
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ Correction timeout, orijinal soru")
            return question
        except Exception as e:
            logger.error(f"❌ Correction hatası: {e}")
            return question

    async def _get_vector_context_ultra_fast(self, question: str) -> str:
        """Ultra hızlı vector search"""
        try:
            if not self.astra_collection:
                return "Vector arama mevcut değil"
            
            # Query optimization SKIP - direkt ara
            query_embedding = self.get_embedding(question)
            
            # Hızlı arama (limit 2)
            results = self.astra_collection.find(
                {},
                sort={"$vector": query_embedding},
                limit=2  # Daha az sonuç = daha hızlı
            )
            
            docs = list(results)
            
            if not docs:
                return "İlgili doküman bulunamadı"
            
            # Kompakt context
            context = ""
            for doc in docs:
                file_path = doc.get('metadata', {}).get('file_path', 'Kaynak')
                content = str(doc.get('content', doc.get('text', '')))[:300]  # Daha kısa
                context += f"{file_path}: {content}\n"
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Ultra fast vector hatası: {e}")
            return "Vector arama hatası"

    async def _get_csv_context_ultra_fast(self, question: str) -> str:
        """Ultra hızlı CSV analiz"""
        try:
            if self.csv_data is None or not self.llm_csv_agent:
                return "CSV analizi mevcut değil"
            
            # Basit filtreleme
            question_lower = question.lower()
            
            # Sadece istihdam/maaş keyword'ü varsa CSV analizi yap
            csv_keywords = ["istihdam", "maaş", "gelir", "çalışma", "iş"]
            if not any(kw in question_lower for kw in csv_keywords):
                return "Bu soru için CSV analizi gerekli değil"
            
            # Hızlı sample data
            sample_data = self.csv_data.head(10).to_string(index=False)
            
            result = await asyncio.wait_for(
                self.llm_csv_agent.ainvoke(
                    self.csv_agent_prompt.format(question=question, csv_data=sample_data)
                ),
                timeout=5.0
            )
            
            return result.content.strip()
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ CSV timeout")
            return "CSV analizi zaman aşımı"
        except Exception as e:
            logger.error(f"❌ Ultra fast CSV hatası: {e}")
            return "CSV analizi hatası"

    async def _generate_final_response_ultra_fast(self, question: str, context1: str, context2: str, history: str = "") -> str:
        """Ultra hızlı final yanıt"""
        try:
            if not self.llm_final:
                return "Sistem geçici olarak kullanılamıyor. Lütfen tekrar deneyin."
            
            result = await asyncio.wait_for(
                self.llm_final.ainvoke(
                    self.final_prompt.format(
                        question=question,
                        context1=context1,
                        context2=context2,
                        history=history[:200]  # History'yi kısalt
                    )
                ),
                timeout=8.0
            )
            
            return result.content.strip()
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ Final response timeout")
            return "Yanıt oluşturma zaman aşımı. Lütfen tekrar deneyin."
        except Exception as e:
            logger.error(f"❌ Final response hatası: {e}")
            return "Yanıt oluşturulurken hata oluştu."

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Hızlı kaynak çıkarma"""
        sources = []
        
        if context1 and "bulunamadı" not in context1 and "hatası" not in context1:
            sources.append("Tercih Rehberi Dokümanları")
        
        if context2 and "gerekli değil" not in context2 and "hatası" not in context2:
            sources.append("Cumhurbaşkanlığı UNİ-VERİ veritabanı (2024)")
        
        return sources

    async def test_all_connections(self) -> Dict[str, str]:
        """Hızlı connection test"""
        results = {}
        
        # OpenAI test
        try:
            test_embedding = self.get_embedding("test")
            results["OpenAI"] = f"✅ Bağlı ({len(test_embedding)} boyut)"
        except Exception as e:
            results["OpenAI"] = f"❌ Hata: {str(e)[:30]}"
        
        # AstraDB test
        try:
            if self.astra_collection:
                test_results = list(self.astra_collection.find({}, limit=1))
                results["AstraDB"] = f"✅ Bağlı ({len(test_results)} test)"
            else:
                results["AstraDB"] = "❌ Bağlantı yok"
        except Exception as e:
            results["AstraDB"] = f"❌ Hata: {str(e)[:30]}"
        
        # CSV test
        results["CSV"] = f"✅ {len(self.csv_data)} satır" if self.csv_data is not None else "❌ Yok"
        
        return results
