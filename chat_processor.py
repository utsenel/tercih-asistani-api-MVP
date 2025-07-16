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
import time
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
    Güncellenmiş processor - Smart Evaluator-Corrector ile
    """
    
    def __init__(self):
        # YENİ: Smart Evaluator-Corrector
        self.llm_smart_evaluator_corrector = None
        
        # KALAN LLM'LER - eski evaluation ve correction kaldırıldı
        self.llm_search_optimizer = None
        self.llm_csv_agent = None
        self.llm_final = None
        
        # Native Astrapy components
        self.openai_client = None
        self.astra_database = None
        self.astra_collection = None
        
        self.csv_data = None
        self.memory = ConversationMemory() 
        
        # YENİ PROMPT
        self.smart_evaluator_corrector_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.SMART_EVALUATOR_CORRECTOR
        )
        
        # KALAN PROMPT'LAR - değişmedi
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

    def _get_recent_history(self, session_id: str, limit: int = 4) -> str:
        """Son N mesajı al - Smart Evaluator için"""
        try:
            if not self.memory:
                return ""
            
            # Memory'den son mesajları al
            full_history = self.memory.get_history(session_id, limit=limit)
            
            if not full_history:
                return ""
            
            # Son 2-3 mesaj çiftini al (user-assistant pairs)
            lines = full_history.strip().split('\n')
            recent_lines = lines[-4:] if len(lines) >= 4 else lines  # Son 4 satır (2 mesaj çifti)
            
            recent_history = '\n'.join(recent_lines)
            
            logger.info(f"📜 Recent history alındı: {len(recent_history)} karakter")
            logger.info(f"   İçerik: '{recent_history[:100]}...'")
            
            return recent_history
            
        except Exception as e:
            logger.error(f"❌ Recent history alma hatası: {e}")
            return ""

    async def initialize(self):
        """Güncellenmiş başlatma - Smart Evaluator ile"""
        try:
            logger.info("🚀 TercihAsistaniProcessor başlatılıyor...")
            
            # API Key kontrolü
            self._check_api_keys()
            
            # OpenAI client'ı başlat
            self._initialize_openai_client()
            
            # YENİ LLM'leri sıralı başlat
            await self._initialize_llms_new()
            
            # AstraDB bağlantısını native API ile başlat
            await self._initialize_astradb_native()
            
            # CSV verilerini yükle
            await self._initialize_csv()
                
            logger.info("✅ TercihAsistaniProcessor başarıyla başlatıldı")
            
        except Exception as e:
            logger.error(f"❌ Initialization hatası: {e}")
            raise

    def _check_api_keys(self):
        """API anahtarlarını kontrol et"""
        keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "ASTRA_DB_TOKEN": os.getenv("ASTRA_DB_TOKEN"),
            "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT")
        }
        
        logger.info("🔑 API Key durumu:")
        for key, value in keys.items():
            status = "✅ Set" if value else "❌ Missing"
            logger.info(f"   {key}: {status}")
            
        # Critical keys check
        if not keys["OPENAI_API_KEY"]:
            raise ValueError("OPENAI_API_KEY zorunlu!")

    def _initialize_openai_client(self):
        """OpenAI client'ı başlat"""
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ OpenAI client başlatıldı")
        except Exception as e:
            logger.error(f"❌ OpenAI client hatası: {e}")
            raise

    async def _initialize_llms_new(self):
        """YENİ LLM başlatma - Smart Evaluator ile"""
        llm_configs = {
            "smart_evaluator_corrector": LLMConfigs.SMART_EVALUATOR_CORRECTOR,
            "search_optimizer": LLMConfigs.SEARCH_OPTIMIZER,
            "csv_agent": LLMConfigs.CSV_AGENT,
            "final": LLMConfigs.FINAL_RESPONSE
        }
        
        for name, config in llm_configs.items():
            try:
                llm = LLMFactory.create_llm(config)
                setattr(self, f"llm_{name}", llm)
                logger.info(f"✅ {name} LLM başarılı: {config.model}")
                
            except Exception as e:
                logger.error(f"❌ {name} LLM hatası: {e}")
                
                # Critical LLM'ler için fallback
                if name in ["smart_evaluator_corrector", "csv_agent", "final"]:
                    logger.warning(f"🔄 {name} için OpenAI fallback...")
                    fallback_config = LLMConfigs.FINAL_RESPONSE  # OpenAI model
                    try:
                        llm = LLMFactory.create_llm(fallback_config)
                        setattr(self, f"llm_{name}", llm)
                        logger.info(f"✅ {name} fallback başarılı")
                    except Exception as fb_error:
                        logger.error(f"❌ {name} fallback hatası: {fb_error}")
                        setattr(self, f"llm_{name}", None)

    async def _initialize_astradb_native(self):
        """AstraDB native API ile bağlantı"""
        try:
            logger.info("🔌 AstraDB native API bağlantısı başlatılıyor...")
            
            # Astra client oluştur
            astra_client = DataAPIClient(DatabaseSettings.ASTRA_DB_TOKEN)
            
            # Database bağlantısı
            self.astra_database = astra_client.get_database(
                DatabaseSettings.ASTRA_DB_API_ENDPOINT
            )
            
            # Collection al
            collection_name = DatabaseSettings.ASTRA_DB_COLLECTION
            self.astra_collection = self.astra_database.get_collection(collection_name)
            
            logger.info(f"✅ AstraDB native bağlantısı başarılı - Collection: {collection_name}")
            
            # Test sorgusu
            test_results = list(self.astra_collection.find({}, limit=1))
            logger.info(f"✅ Test sorgusu başarılı: {len(test_results)} doküman bulundu")
            
        except Exception as e:
            logger.error(f"❌ AstraDB native bağlantı hatası: {e}")
            self.astra_database = None
            self.astra_collection = None

    async def _initialize_csv(self):
        """CSV verilerini güvenli yükle"""
        try:
            csv_path = DatabaseSettings.CSV_FILE_PATH
            
            if not csv_path or not os.path.exists(csv_path):
                logger.warning(f"⚠️ CSV dosyası bulunamadı: {csv_path}")
                self.csv_data = None
                return
            
            self.csv_data = pd.read_csv(csv_path)
            
            # Veri validasyonu
            if self.csv_data.empty:
                logger.warning("⚠️ CSV dosyası boş")
                self.csv_data = None
                return
            
            # Gerekli kolonların varlığını kontrol et
            required_cols = ['bolum_adi', 'gosterge_id']
            missing_cols = [col for col in required_cols if col not in self.csv_data.columns]
            
            if missing_cols:
                logger.error(f"❌ CSV'de eksik kolonlar: {missing_cols}")
                self.csv_data = None
                return
            
            logger.info(f"✅ CSV verisi yüklendi: {len(self.csv_data)} satır, {len(self.csv_data.columns)} kolon")
            
        except Exception as e:
            logger.error(f"❌ CSV yükleme hatası: {e}")
            self.csv_data = None

    async def _smart_evaluate_and_correct(self, message: str, session_id: str) -> Dict[str, str]:
        """YENİ: Smart Evaluator-Corrector fonksiyonu"""
        try:
            smart_start = time.time()
            
            if not self.llm_smart_evaluator_corrector:
                logger.warning("⚠️ Smart Evaluator-Corrector LLM mevcut değil, fallback")
                return {
                    "status": "UYGUN",
                    "enhanced_question": message
                }
            
            # Son birkaç mesajı al
            recent_history = self._get_recent_history(session_id, limit=4)
            
            logger.info(f"🧠 SMART EVALUATOR-CORRECTOR başlatılıyor:")
            logger.info(f"   📝 Orijinal mesaj: '{message[:50]}...'")
            logger.info(f"   📜 History: {len(recent_history)} karakter")
            
            # Smart Evaluator-Corrector'a gönder
            result = await self.llm_smart_evaluator_corrector.ainvoke(
                self.smart_evaluator_corrector_prompt.format(
                    question=message,
                    history=recent_history
                )
            )
            
            response = result.content.strip()
            smart_time = time.time() - smart_start
            
            logger.info(f"🤖 Smart Evaluator-Corrector raw response ({smart_time:.2f}s):")
            logger.info(f"   📄 Raw output: '{response[:150]}...'")
            
            # Response'u parse et
            try:
                # STATUS ve ENHANCED_QUESTION'u extract et
                status_match = re.search(r'STATUS:\s*(\w+)', response)
                question_match = re.search(r'ENHANCED_QUESTION:\s*(.+)', response, re.DOTALL)
                
                if status_match and question_match:
                    status = status_match.group(1).strip()
                    enhanced_question = question_match.group(1).strip()
                    
                    logger.info(f"✅ PARSE BAŞARILI:")
                    logger.info(f"   📊 Status: {status}")
                    logger.info(f"   📝 Enhanced Q: '{enhanced_question[:80]}...'")
                    
                    return {
                        "status": status,
                        "enhanced_question": enhanced_question
                    }
                else:
                    logger.warning("⚠️ Parse başarısız - format hatası")
                    logger.warning(f"   Status match: {bool(status_match)}")
                    logger.warning(f"   Question match: {bool(question_match)}")
                    
                    # Fallback parsing
                    if "UYGUN" in response.upper():
                        return {"status": "UYGUN", "enhanced_question": message}
                    elif "SELAMLAMA" in response.upper():
                        return {"status": "SELAMLAMA", "enhanced_question": message}
                    else:
                        return {"status": "KAPSAM_DIŞI", "enhanced_question": message}
                        
            except Exception as parse_error:
                logger.error(f"❌ Parse hatası: {parse_error}")
                return {"status": "UYGUN", "enhanced_question": message}
            
        except Exception as e:
            smart_time = time.time() - smart_start
            logger.error(f"❌ Smart Evaluator-Corrector hatası ({smart_time:.2f}s): {e}")
            return {"status": "UYGUN", "enhanced_question": message}

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """YENİ akış ile mesaj işleme"""
        start_time = time.time()
        
        try:
            logger.info(f"📨 Mesaj işleniyor - Session: {session_id}")
            logger.info(f"📝 Gelen mesaj: '{message[:100]}...' ({len(message)} karakter)")
            
            # Adım 1: YENİ Smart Evaluator-Corrector
            smart_start = time.time()
            smart_result = await self._smart_evaluate_and_correct(message, session_id)
            smart_time = time.time() - smart_start
            
            status = smart_result["status"]
            enhanced_question = smart_result["enhanced_question"]
            
            logger.info(f"⏱️ Smart Evaluator-Corrector süresi: {smart_time:.2f}s")
            logger.info(f"📊 Status: {status}")
            logger.info(f"📝 Enhanced Question: '{enhanced_question[:100]}...'")
            
            # Adım 2: Koşullu yönlendirme
            if status == "KAPSAM_DIŞI":
                total_time = time.time() - start_time
                logger.info(f"🚫 Kapsam dışı soru - Toplam süre: {total_time:.2f}s")
                return {
                    "response": MessageSettings.ERROR_EXPERTISE_OUT,
                    "sources": []
                }
            
            if status == "SELAMLAMA":
                total_time = time.time() - start_time
                logger.info(f"👋 Selamlama algılandı - Toplam süre: {total_time:.2f}s")
                return {
                    "response": "Merhaba! Ben bir üniversite tercih asistanıyım. Size YKS tercihleri, bölüm seçimi, kariyer planlaması konularında yardımcı olabilirim. Hangi konuda bilgi almak istiyorsunuz?",
                    "sources": []
                }
            
            # Adım 3: PARALEL İŞLEMLER - Enhanced question ile
            parallel_start = time.time()
            logger.info("🔄 Paralel işlemler başlatılıyor...")
            
            # Task'ları oluştur
            vector_task = asyncio.create_task(
                self._get_vector_context_native(enhanced_question)
            )
            csv_task = asyncio.create_task(
                self._get_csv_context_safe(enhanced_question)
            )
            
            # Paralel yürütme
            try:
                context1, context2 = await asyncio.gather(
                    vector_task, 
                    csv_task, 
                    return_exceptions=True
                )
            except Exception as e:
                logger.error(f"❌ Paralel işleme genel hatası: {e}")
                context1 = "Vector arama başarısız"
                context2 = "CSV analizi başarısız"
            
            parallel_time = time.time() - parallel_start
            logger.info(f"⏱️ Paralel işlemler toplam süresi: {parallel_time:.2f}s")
            
            # Exception'ları handle et
            if isinstance(context1, Exception):
                logger.error(f"❌ Vector context hatası: {context1}")
                context1 = "Vector arama başarısız"
                
            if isinstance(context2, Exception):
                logger.error(f"❌ CSV context hatası: {context2}")
                context2 = "CSV analizi başarısız"
            
            # Context detaylarını logla
            logger.info(f"📄 CONTEXT1 (Vector) - {len(context1)} karakter")
            logger.info(f"📊 CONTEXT2 (CSV) - {len(context2)} karakter")
            
            # Adım 4: Memory'den geçmiş al
            memory_start = time.time()
            conversation_history = self.memory.get_history(session_id)
            memory_time = time.time() - memory_start
            logger.info(f"🧠 Memory geçmişi alındı ({memory_time:.3f}s): {len(conversation_history)} karakter")
            
            # Adım 5: Final yanıt oluşturma - Enhanced question ile
            final_start = time.time()
            final_response = await self._generate_final_response_safe(
                question=enhanced_question,  # Enhanced question kullan
                context1=context1,
                context2=context2,
                history=conversation_history
            )
            final_time = time.time() - final_start
            logger.info(f"⏱️ Final response süresi: {final_time:.2f}s")
            logger.info(f"✅ Final yanıt: {len(final_response)} karakter")

            # Memory'ye kaydet - orijinal mesajı kaydet
            memory_save_start = time.time()
            self.memory.add_message(session_id, "user", message)  # Orijinal mesaj
            self.memory.add_message(session_id, "assistant", final_response)
            memory_save_time = time.time() - memory_save_start
            logger.info(f"💾 Memory kayıt tamamlandı ({memory_save_time:.3f}s)")

            # PERFORMANS RAPORU
            total_time = time.time() - start_time
            logger.info(f"📈 PERFORMANS RAPORU:")
            logger.info(f"   🧠 Smart Evaluator-Corrector: {smart_time:.2f}s")
            logger.info(f"   🔄 Paralel İşlemler: {parallel_time:.2f}s")
            logger.info(f"   🧠 Memory: {memory_time:.3f}s")
            logger.info(f"   🎯 Final Response: {final_time:.2f}s")
            logger.info(f"   💾 Memory Save: {memory_save_time:.3f}s")
            logger.info(f"   🎉 TOPLAM: {total_time:.2f}s")
            
            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2),
                "metadata": {
                    "processing_time": round(total_time, 2),
                    "smart_evaluator_time": round(smart_time, 2),
                    "parallel_time": round(parallel_time, 2),
                    "enhanced_question": enhanced_question,
                    "original_question": message,
                    "status": status
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Mesaj işleme genel hatası ({total_time:.2f}s): {e}")
            return {
                "response": MessageSettings.ERROR_GENERAL,
                "sources": [],
                "metadata": {"error": str(e), "processing_time": round(total_time, 2)}
            }

    async def _get_vector_context_native(self, question: str) -> str:
        """Native AstraDB ile vector arama"""
        try:
            vector_start = time.time()
            
            if not self.astra_collection:
                logger.warning("❌ Astra collection mevcut değil")
                return "Vector arama mevcut değil"
            
            logger.info(f"🔍 Native vector arama başlatılıyor: {question[:50]}...")
            
            # Search query optimize et
            search_text = question
            if self.llm_search_optimizer:
                try:
                    optimized_query = await self.llm_search_optimizer.ainvoke(
                        self.search_optimizer_prompt.format(question=question)
                    )
                    search_text = optimized_query.content.strip()
                    logger.info(f"✨ Optimize edilmiş sorgu: {search_text[:80]}...")
                except Exception as e:
                    logger.warning(f"⚠️ Sorgu optimizasyonu başarısız: {e}")
            
            # Embedding oluştur
            try:
                query_embedding = self.get_embedding(search_text)
                logger.info(f"✅ Query embedding oluşturuldu: {len(query_embedding)} boyut")
            except Exception as e:
                logger.error(f"❌ Embedding oluşturma hatası: {e}")
                return "Embedding oluşturulamadı"
            
            # Native vector search
            try:
                search_results = self.astra_collection.find(
                    {},
                    sort={"$vector": query_embedding},
                    limit=VectorConfig.SIMILARITY_TOP_K,
                    projection={
                        "$vectorize": 1,
                        "text": 1, 
                        "content": 1, 
                        "page_content": 1,
                        "body": 1,
                        "document": 1,
                        "metadata": 1,
                        "source": 1,
                        "file_path": 1,
                        "_id": 1
                    }
                )
                
                docs = list(search_results)
                logger.info(f"📄 Bulunan doküman sayısı: {len(docs)}")
                
                if not docs:
                    logger.warning("❌ Hiç doküman bulunamadı")
                    return "İlgili doküman bulunamadı"
                
                # Doküman içeriklerini birleştir
                context_parts = []
                total_chars = 0
                
                for i, doc in enumerate(docs):
                    try:
                        # İçerik al
                        content = None
                        content_source = None
                        
                        if '$vectorize' in doc and doc['$vectorize']:
                            content = str(doc['$vectorize']).strip()
                            content_source = '$vectorize'
                        else:
                            # Fallback fields
                            possible_content_fields = ['text', 'content', 'page_content', 'body', 'document']
                            for field in possible_content_fields:
                                if field in doc and doc[field]:
                                    content = str(doc[field]).strip()
                                    content_source = field
                                    break
                        
                        if not content:
                            continue
                        
                        # Kaynak bilgisi
                        source = "Bilinmeyen kaynak"
                        if 'metadata' in doc and isinstance(doc['metadata'], dict):
                            metadata = doc['metadata']
                            source = metadata.get('source', metadata.get('file_path', metadata.get('filename', source)))
                        elif 'source' in doc:
                            source = doc['source']
                        elif 'file_path' in doc:
                            source = doc['file_path']
                        
                        # İçeriği kısalt
                        if len(content) > 800:
                            content = content[:800] + "..."
                        
                        # Kaynak formatını düzelt
                        if isinstance(source, str):
                            source_name = source.split('/')[-1] if '/' in source else source
                            if any(char in source_name for char in ['Ä°', 'ZÃ', 'Ã', 'Â']):
                                source_name = "İZÜ YKS Tercih Rehberi.pdf"
                            if not source_name or source_name == "Bilinmeyen kaynak":
                                source_name = "Tercih Rehberi"
                        else:
                            source_name = "Rehber Dokümanı"
                        
                        context_parts.append(f"**Kaynak**: {source_name}\n**İçerik**: {content}")
                        total_chars += len(content)
                        
                        logger.info(f"✅ Doküman {i+1} işlendi: {source_name} - {len(content)} karakter")
                        
                        if total_chars > 2000:
                            break
                            
                    except Exception as doc_error:
                        logger.error(f"❌ Doküman {i+1} işleme hatası: {doc_error}")
                        continue
                
                if not context_parts:
                    logger.error("❌ Hiçbir doküman işlenemedi!")
                    return "Dokümanlar işlenemedi"
                
                final_context = "\n\n".join(context_parts)
                vector_time = time.time() - vector_start
                
                logger.info(f"✅ NATIVE VECTOR ARAMA TAMAMLANDI ({vector_time:.2f}s):")
                logger.info(f"   📄 İşlenen doküman: {len(context_parts)} adet")
                logger.info(f"   📝 Toplam context: {len(final_context)} karakter")
                
                return final_context
                    
            except Exception as search_error:
                logger.error(f"❌ Vector arama hatası: {search_error}")
                return "Vector arama başarısız"
            
        except Exception as e:
            vector_time = time.time() - vector_start
            logger.error(f"❌ Vector context genel hatası ({vector_time:.2f}s): {e}")
            return "Vector arama genel hatası"

    async def _get_csv_context_safe(self, question: str) -> str:
        """CSV analiz - Enhanced question ile"""
        try:
            csv_start = time.time()
            
            if self.csv_data is None:
                logger.info("❌ CSV verileri mevcut değil")
                return "CSV verileri mevcut değil"
    
            question_lower = question.lower()
            logger.info(f"🔍 CSV analizi: '{question_lower[:50]}...'")
            
            # CSV anahtar kelimesi kontrolü
            csv_keywords = [
                "istihdam", "maaş", "gelir", "sektör", "firma", "çalışma", "iş", 
                "girişim", "başlama", "oran", "yüzde", "istatistik", "veri",
                "bilgisayar", "mühendislik", "tıp", "hukuk", "ekonomi", "matematik",
                "fizik", "kimya", "makine", "elektrik", "endüstri"
            ]
            
            csv_required = any(keyword in question_lower for keyword in csv_keywords)
            logger.info(f"🔍 CSV Keywords check: {csv_required}")
            
            if not csv_required:
                logger.info("⚡ CSV analizi atlandı")
                return "CSV analizi gerekli değil"
    
            # Bölüm adını bul
            bolum_adi = None
            
            for bolum in self.csv_data['bolum_adi'].unique():
                if bolum.lower() in question_lower:
                    bolum_adi = bolum
                    break
            
            if not bolum_adi:
                for bolum in self.csv_data['bolum_adi'].unique():
                    bolum_words = bolum.lower().split()
                    if any(word in question_lower for word in bolum_words if len(word) > 3):
                        bolum_adi = bolum
                        break
    
            # Spesifik bölüm analizi
            if bolum_adi:
                logger.info(f"📋 Spesifik bölüm analizi: {bolum_adi}")
                
                filtered = self.csv_data[self.csv_data['bolum_adi'] == bolum_adi]
                
                if filtered.empty:
                    return f"{bolum_adi} için veri bulunamadı"
                
                # Metrik sütunlarını belirle
                metrik_cols = ["istihdam_orani", "girisimcilik_orani", "ortalama_calisma_suresi_ay"]
                
                if any(word in question_lower for word in ["istihdam", "çalışma", "iş"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if "istihdam" in col])
                    
                if any(word in question_lower for word in ["maaş", "gelir", "salary"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("maas_")])
                    
                if any(word in question_lower for word in ["sektör", "sector"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("sektor_")])
                    
                if any(word in question_lower for word in ["firma", "şirket"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("firma_")])
                
                metrik_cols = list(dict.fromkeys(metrik_cols))[:30]
                
                selected_cols = ['bolum_adi', 'gosterge_id'] + metrik_cols
                csv_snippet = filtered[selected_cols].to_string(index=False)
                
            else:
                # Genel analiz
                logger.info("📈 Genel CSV analizi")
                top_bolumler = self.csv_data.nlargest(5, 'istihdam_orani')
                sample_cols = ['bolum_adi', 'istihdam_orani', 'girisimcilik_orani', 'ortalama_calisma_suresi_ay']
                csv_snippet = top_bolumler[sample_cols].to_string(index=False)
    
            # CSV Agent'a sor
            if self.llm_csv_agent:
                try:
                    result = await self.llm_csv_agent.ainvoke(
                        self.csv_agent_prompt.format(
                            question=question,
                            csv_data=csv_snippet[:2000]
                        )
                    )
                    analysis = result.content.strip()
                    
                    if len(analysis) < 20:
                        analysis = f"CSV analizi tamamlandı. {bolum_adi or 'İlgili bölümler'} için temel veriler: {csv_snippet[:200]}..."
                        
                except Exception as agent_error:
                    logger.error(f"❌ CSV Agent hatası: {agent_error}")
                    analysis = f"CSV verisi bulundu: {csv_snippet[:300]}..."
            else:
                analysis = f"CSV verisi: {csv_snippet[:300]}..."
    
            csv_time = time.time() - csv_start
            logger.info(f"⏱️ CSV analizi süresi: {csv_time:.2f}s")
            
            return analysis
    
        except Exception as e:
            csv_time = time.time() - csv_start
            logger.error(f"❌ CSV analiz hatası ({csv_time:.2f}s): {e}")
            return "CSV analizi hatası"
            
    async def _generate_final_response_safe(self, question: str, context1: str, context2: str, history: str = "") -> str:
        """Final yanıt oluşturma"""
        try:
            if not self.llm_final:
                logger.error("❌ Final LLM mevcut değil!")
                return "Yanıt oluşturma servisi geçici olarak kullanılamıyor."
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2,
                    history=history
                )
            )
            
            final_response = result.content.strip()
            logger.info(f"✅ Final response oluşturuldu: {len(final_response)} karakter")
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Final yanıt hatası: {e}")
            return "Yanıt oluşturulurken hata oluştu."

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Kaynak çıkarma"""
        sources = []
        
        # Vector context kontrolü
        if context1 and len(context1.strip()) > 50:
            error_keywords = ["bulunamadı", "başarısız", "mevcut değil", "hata"]
            has_error = any(keyword in context1.lower() for keyword in error_keywords)
            
            if not has_error:
                if "İZÜ" in context1 or "tercih rehberi" in context1.lower():
                    sources.append(MessageSettings.SOURCES["IZU_GUIDE"])
                elif "yök" in context1.lower():
                    sources.append(MessageSettings.SOURCES["YOK_REPORT"])
                else:
                    sources.append(MessageSettings.SOURCES["IZU_GUIDE"])
        
        # CSV context kontrolü
        if context2 and len(context2.strip()) > 50:
            csv_error_keywords = ["mevcut değil", "hata", "başarısız", "gerekli değil"]
            has_csv_error = any(keyword in context2.lower() for keyword in csv_error_keywords)
            
            if not has_csv_error:
                csv_success_indicators = ["analiz", "oran", "veri", "bölüm", "istihdam", "maaş", "%"]
                has_csv_content = any(indicator in context2.lower() for indicator in csv_success_indicators)
                
                if has_csv_content:
                    sources.append(MessageSettings.SOURCES["UNIVERI_DB"])
        
        if not sources:
            sources.append(MessageSettings.SOURCES["GENERAL"])
        
        return list(dict.fromkeys(sources))

    async def test_all_connections(self) -> Dict[str, str]:
        """Bağlantı testleri"""
        logger.info("🧪 TÜM BAĞLANTILAR TEST EDİLİYOR...")
        results = {}
        
        # OpenAI Client test
        try:
            if self.openai_client:
                test_embedding = self.get_embedding("test")
                results["OpenAI Client"] = f"✅ Bağlı ({len(test_embedding)} boyut)"
            else:
                results["OpenAI Client"] = "❌ Client başlatılmadı"
        except Exception as e:
            results["OpenAI Client"] = f"❌ Hata: {str(e)[:50]}"
        
        # LLM testleri
        llm_tests = [
            ("Smart Evaluator-Corrector", self.llm_smart_evaluator_corrector),
            ("Search Optimizer", self.llm_search_optimizer),
            ("CSV Agent", self.llm_csv_agent),
            ("Final Response", self.llm_final)
        ]
        
        for name, llm in llm_tests:
            try:
                if llm:
                    await llm.ainvoke("Test")
                    results[name] = "✅ Bağlı"
                else:
                    results[name] = "❌ Model yüklenmedi"
            except Exception as e:
                results[name] = f"❌ Hata: {str(e)[:50]}"
        
        # AstraDB test
        try:
            if self.astra_collection:
                test_results = list(self.astra_collection.find({}, limit=1))
                results["AstraDB Native"] = f"✅ Bağlı ({len(test_results)} doküman)"
            else:
                results["AstraDB Native"] = "❌ Collection başlatılmadı"
        except Exception as e:
            results["AstraDB Native"] = f"❌ Hata: {str(e)[:50]}"
        
        # CSV test
        try:
            if self.csv_data is not None:
                results["CSV"] = f"✅ Yüklü ({len(self.csv_data)} satır)"
            else:
                results["CSV"] = "❌ Yüklenmedi"
        except Exception as e:
            results["CSV"] = f"❌ Hata: {str(e)[:50]}"
        
        # Memory test
        try:
            self.memory.add_message("test_connection", "user", "test")
            history = self.memory.get_history("test_connection")
            if history:
                results["Memory"] = "✅ Redis bağlı"
            else:
                results["Memory"] = "⚠️ Memory çalışıyor ama boş"
        except Exception as e:
            results["Memory"] = f"❌ Hata: {str(e)[:50]}"
        
        return results
