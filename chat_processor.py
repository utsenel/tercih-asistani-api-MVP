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
    Paralel işleme ve detaylı logging ile geliştirilmiş processor
    """
    
    def __init__(self):
        self.llm_evaluation = None
        self.llm_correction = None
        self.llm_search_optimizer = None
        self.llm_csv_agent = None
        self.llm_final = None
        
        # Native Astrapy components
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
        """Gelişmiş hata yönetimi ile başlatma"""
        try:
            logger.info("🚀 TercihAsistaniProcessor başlatılıyor...")
            
            # API Key kontrolü
            self._check_api_keys()
            
            # OpenAI client'ı başlat
            self._initialize_openai_client()
            
            # LLM'leri sıralı başlat (fallback ile)
            await self._initialize_llms()
            
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

    async def _initialize_llms(self):
        """Fallback stratejisi ile LLM'leri başlat"""
        llm_configs = {
            "evaluation": LLMConfigs.EVALUATION,
            "correction": LLMConfigs.CORRECTION, 
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
                if name in ["csv_agent", "final"]:
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

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Gelişmiş paralel işleme ve detaylı logging ile mesaj işleme"""
        start_time = time.time()
        
        try:
            logger.info(f"📨 Mesaj işleniyor - Session: {session_id}")
            logger.info(f"📝 Gelen mesaj: '{message[:100]}...' ({len(message)} karakter)")
            
            # Adım 1: Soru uygunluk değerlendirmesi
            eval_start = time.time()
            evaluation_result = await self._evaluate_question_safe(message)
            eval_time = time.time() - eval_start
            logger.info(f"⏱️ Evaluation süresi: {eval_time:.2f}s")
            
            # Adım 2: Koşullu yönlendirme
            if evaluation_result == "Uzmanlık dışı soru":
                total_time = time.time() - start_time
                logger.info(f"🚫 Uzmanlık dışı soru - Toplam süre: {total_time:.2f}s")
                return {
                    "response": MessageSettings.ERROR_EXPERTISE_OUT,
                    "sources": []
                }
            
            if evaluation_result == "SELAMLAMA":
                total_time = time.time() - start_time
                logger.info(f"👋 Selamlama algılandı - Toplam süre: {total_time:.2f}s")
                return {
                    "response": "Merhaba! Ben bir üniversite tercih asistanıyım. Size YKS tercihleri, bölüm seçimi, kariyer planlaması konularında yardımcı olabilirim. Hangi konuda bilgi almak istiyorsunuz?",
                    "sources": []
                }
            
            # Adım 3: Soru düzeltme
            correction_start = time.time()
            corrected_question = await self._correct_question_safe(message)
            correction_time = time.time() - correction_start
            logger.info(f"⏱️ Correction süresi: {correction_time:.2f}s")
            logger.info(f"📝 Düzeltilmiş soru: '{corrected_question}'")
            
            # Adım 4: PARALEL İŞLEMLER - Task'lar ile force et
            parallel_start = time.time()
            logger.info("🔄 Paralel işlemler başlatılıyor...")
            
            # Task'ları oluştur (gerçek paralel çalışma için)
            vector_task = asyncio.create_task(
                self._get_vector_context_native(corrected_question)
            )
            csv_task = asyncio.create_task(
                self._get_csv_context_safe(corrected_question)
            )
            
            # Paralel yürütme (exception handling ile)
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
            logger.info(f"📄 CONTEXT1 (Vector) - {len(context1)} karakter:")
            logger.info(f"   İlk 200 karakter: '{context1[:500]}...'")
            
            logger.info(f"📊 CONTEXT2 (CSV) - {len(context2)} karakter:")
            logger.info(f"   İlk 200 karakter: '{context2[:200]}...'")
            
            # Adım 5: Memory'den geçmiş al
            memory_start = time.time()
            conversation_history = self.memory.get_history(session_id)
            memory_time = time.time() - memory_start
            logger.info(f"🧠 Memory geçmişi alındı ({memory_time:.3f}s): {len(conversation_history)} karakter")
            if conversation_history:
                logger.info(f"   Geçmiş özet: '{conversation_history[:100]}...'")
            
            # Adım 6: Final yanıt oluşturma
            final_start = time.time()
            final_response = await self._generate_final_response_safe(
                question=corrected_question,
                context1=context1,
                context2=context2,
                history=conversation_history
            )
            final_time = time.time() - final_start
            logger.info(f"⏱️ Final response süresi: {final_time:.2f}s")
            logger.info(f"✅ Final yanıt: {len(final_response)} karakter")
            logger.info(f"   İlk 150 karakter: '{final_response[:150]}...'")

            # Memory'ye kaydet
            memory_save_start = time.time()
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", final_response)
            memory_save_time = time.time() - memory_save_start
            logger.info(f"💾 Memory kayıt tamamlandı ({memory_save_time:.3f}s)")

            # PERFORMANS RAPORU
            total_time = time.time() - start_time
            logger.info(f"📈 PERFORMANS RAPORU:")
            logger.info(f"   ⚡ Evaluation: {eval_time:.2f}s")
            logger.info(f"   ✏️  Correction: {correction_time:.2f}s")
            logger.info(f"   🔄 Paralel İşlemler: {parallel_time:.2f}s")
            logger.info(f"   🧠 Memory: {memory_time:.3f}s")
            logger.info(f"   🎯 Final Response: {final_time:.2f}s")
            logger.info(f"   💾 Memory Save: {memory_save_time:.3f}s")
            logger.info(f"   🎉 TOPLAM: {total_time:.2f}s")
            
            # Performance warning
            if total_time > 15:
                logger.warning(f"⚠️ Yavaş yanıt: {total_time:.2f}s > 15s!")
            elif total_time > 8:
                logger.info(f"⚠️ Ortalama yanıt: {total_time:.2f}s")
            else:
                logger.info(f"✅ Hızlı yanıt: {total_time:.2f}s")

            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2),
                "metadata": {
                    "processing_time": round(total_time, 2),
                    "parallel_time": round(parallel_time, 2),
                    "context1_length": len(context1),
                    "context2_length": len(context2)
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

    async def _evaluate_question_safe(self, question: str) -> str:
        """Güvenli soru değerlendirme"""
        try:
            if not self.llm_evaluation:
                logger.warning("⚠️ Evaluation LLM mevcut değil, varsayılan UYGUN")
                return "UYGUN"
                
            result = await self.llm_evaluation.ainvoke(
                self.evaluation_prompt.format(question=question)
            )
            evaluation_result = result.content.strip()
            
            # Detaylı evaluation logging
            logger.info(f"🔍 EVALUATION SONUCU:")
            logger.info(f"   Ham çıktı: '{evaluation_result[:100]}...'")
            
            if "SELAMLAMA" in evaluation_result.upper():
                logger.info(f"   ✅ Karar: SELAMLAMA")
                return "SELAMLAMA"
            elif "UYGUN" in evaluation_result.upper():
                logger.info(f"   ✅ Karar: UYGUN")
                return "UYGUN"
            else:
                logger.info(f"   ❌ Karar: Uzmanlık dışı")
                return "Uzmanlık dışı soru"
                
        except Exception as e:
            logger.error(f"❌ Değerlendirme hatası: {e}")
            return "UYGUN"  # Güvenli varsayılan

    async def _correct_question_safe(self, question: str) -> str:
        """Güvenli soru düzeltme"""
        try:
            if not self.llm_correction:
                logger.warning("⚠️ Correction LLM mevcut değil, orijinal soru döndürülüyor")
                return question
                
            result = await self.llm_correction.ainvoke(
                self.correction_prompt.format(question=question)
            )
            corrected = result.content.strip()
            
            # Detaylı correction logging
            logger.info(f"✏️ CORRECTION SONUCU:")
            logger.info(f"   Orijinal: '{question[:80]}...'")
            logger.info(f"   Düzeltilmiş: '{corrected[:80]}...'")
            
            return corrected
        except Exception as e:
            logger.error(f"❌ Düzeltme hatası: {e}")
            return question

    async def _get_vector_context_native(self, question: str) -> str:
        """Native AstraDB ile vector arama - TAMAMEN YENİ"""
        try:
            vector_start = time.time()
            
            if not self.astra_collection:
                logger.warning("❌ Astra collection mevcut değil")
                return "Vector arama mevcut değil"
            
            logger.info(f"🔍 Native vector arama başlatılıyor: {question[:50]}...")
            
            # Search query optimize et (eğer mümkünse)
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
                    {},  # Empty filter - tüm belgelerden ara
                    sort={"$vector": query_embedding},  # Vector similarity sort
                    limit=VectorConfig.SIMILARITY_TOP_K,
                    projection={"text": 1, "metadata": 1, "_id": 0}  # Sadece gerekli alanlar
                )
                
                # Results'ı listeye çevir
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
                        # Text alanından içeriği al
                        content = doc.get('text', '').strip()
                        
                        # Metadata'dan kaynak bilgisini al
                        metadata = doc.get('metadata', {})
                        source = metadata.get('source', 'Bilinmeyen kaynak')
                        
                        if not content:
                            logger.warning(f"⚠️ Doküman {i+1} boş içerik")
                            continue
                        
                        # İçeriği kısalt
                        if len(content) > 400:
                            content = content[:400] + "..."
                        
                        # Kaynak formatını düzelt
                        if isinstance(source, str):
                            source_name = source.split('/')[-1] if '/' in source else source
                            if 'Ä°ZÃ' in source_name or any(char in source_name for char in ['Ã', 'Â']):
                                source_name = "İZÜ YKS Tercih Rehberi.pdf"
                        else:
                            source_name = "Rehber Dokümanı"
                        
                        context_parts.append(f"**Kaynak**: {source_name}\n**İçerik**: {content}")
                        total_chars += len(content)
                        
                        logger.info(f"✅ Doküman {i+1}: {source_name} - {len(content)} karakter")
                        
                        # Maximum 1200 karakter sınırı
                        if total_chars > 1200:
                            break
                            
                    except Exception as doc_error:
                        logger.warning(f"⚠️ Doküman {i+1} işleme hatası: {doc_error}")
                        continue
                
                if not context_parts:
                    logger.error("❌ Hiçbir doküman işlenemedi!")
                    return "Dokümanlar işlenemedi"
                
                final_context = "\n\n".join(context_parts)
                vector_time = time.time() - vector_start
                
                logger.info(f"✅ NATIVE VECTOR ARAMA TAMAMLANDI ({vector_time:.2f}s):")
                logger.info(f"   📄 İşlenen doküman: {len(context_parts)} adet")
                logger.info(f"   📝 Toplam context: {len(final_context)} karakter")
                logger.info(f"   📄 Context önizleme: '{final_context[:200]}...'")
                
                return final_context
                    
            except Exception as search_error:
                logger.error(f"❌ Vector arama hatası: {search_error}")
                return "Vector arama başarısız"
            
        except Exception as e:
            vector_time = time.time() - vector_start
            logger.error(f"❌ Vector context genel hatası ({vector_time:.2f}s): {e}")
            return "Vector arama genel hatası"
    

    async def _get_csv_context_safe(self, question: str) -> str:
        """CSV analiz - HIZLANDIRILMIŞ VE GÜVENLİ VERSİYON"""
        try:
            csv_start = time.time()
            
            if self.csv_data is None:
                logger.info("❌ CSV verileri mevcut değil")
                return "CSV verileri mevcut değil"

            question_lower = question.lower()
            
            # HIZLI ÖN KONTROL - CSV anahtar kelimesi var mı?
            csv_keywords = [
                "istihdam", "maaş", "gelir", "sektör", "firma", "çalışma", "iş", 
                "girişim", "başlama", "oran", "yüzde", "istatistik", "veri",
                "employment", "salary", "sector", "startup", "rate", "percentage"
            ]
            
            csv_required = any(keyword in question_lower for keyword in csv_keywords)
            
            if not csv_required:
                logger.info("⚡ CSV analizi atlandı - anahtar kelime yok")
                return "CSV analizi gerekli değil - genel rehberlik sorusu"

            logger.info("📊 CSV analizi gerekli - detaylı analiz başlatılıyor")

            # Bölüm adını bul
            bolum_adi = None
            for bolum in self.csv_data['bolum_adi'].unique():
                if bolum.lower() in question_lower:
                    bolum_adi = bolum
                    break

            # Sadece spesifik bölüm sorgusu varsa detaylı analiz
            if bolum_adi:
                logger.info(f"🎯 Spesifik bölüm bulundu: {bolum_adi}")
                
                # Filtreli analiz
                filtered = self.csv_data[self.csv_data['bolum_adi'] == bolum_adi]
                
                if filtered.empty:
                    logger.warning(f"⚠️ {bolum_adi} için veri bulunamadı")
                    return f"{bolum_adi} için veri bulunamadı"
                
                # İlgili metrikleri belirle
                metrik_cols = []
                if any(word in question_lower for word in ["istihdam", "çalışma", "iş", "employment"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if "istihdam" in col])
                if any(word in question_lower for word in ["maaş", "gelir", "salary", "wage"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("maas_")])
                if any(word in question_lower for word in ["sektör", "sector", "alan"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("sektor_")])
                if any(word in question_lower for word in ["firma", "şirket", "company"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if col.startswith("firma_")])
                if any(word in question_lower for word in ["girişim", "startup", "entrepreneur"]):
                    metrik_cols.extend([col for col in self.csv_data.columns if "girisim" in col])
                    
                if not metrik_cols:
                    # Varsayılan metrikler
                    metrik_cols = ["istihdam_orani", "girisimcilik_orani"]
                
                # Küçük veri seti hazırla (ilk 25 metrik)
                selected_cols = ['bolum_adi', 'gosterge_id'] + metrik_cols[:25]
                csv_snippet = filtered[selected_cols].to_string(index=False)
                
                logger.info(f"📋 Seçilen metrikler: {len(metrik_cols)} adet")
                
            else:
                # Genel sorgu - örnek veri ver
                logger.info("📈 Genel CSV sorusu - örnek veri kullanılıyor")
                sample_data = self.csv_data.head(5)[['bolum_adi', 'istihdam_orani', 'girisimcilik_orani']]
                csv_snippet = sample_data.to_string(index=False)

            # CSV Agent'a sor (güvenli fallback ile)
            if self.llm_csv_agent:
                try:
                    result = await self.llm_csv_agent.ainvoke(
                        self.csv_agent_prompt.format(
                            question=question,
                            csv_data=csv_snippet[:1500]  # 1500 karakter sınırı
                        )
                    )
                    analysis = result.content.strip()
                    logger.info(f"✅ CSV Agent analiz tamamlandı")
                except Exception as agent_error:
                    logger.error(f"❌ CSV Agent hatası: {agent_error}")
                    analysis = f"CSV analizi sırasında model hatası oluştu. Ham veri: {csv_snippet[:300]}..."
            else:
                logger.warning("⚠️ CSV Agent LLM mevcut değil - ham veri döndürülüyor")
                analysis = f"CSV analizi için model mevcut değil. İlgili veri bulundu: {csv_snippet[:300]}..."

            csv_time = time.time() - csv_start
            logger.info(f"📄 Analiz özet: '{analysis[:100]}...'")
            logger.info(f"⏱️ Toplam CSV süresi: {csv_time:.2f}s")
            
            return analysis

        except Exception as e:
            csv_time = time.time() - csv_start
            logger.error(f"❌ CSV analiz genel hatası ({csv_time:.2f}s): {e}")
            return "CSV analizi sırasında hata oluştu"
            
    async def _generate_final_response_safe(self, question: str, context1: str, context2: str, history: str = "") -> str:
        """Güvenli final yanıt oluşturma - Detaylı logging"""
        try:
            final_start = time.time()
            
            if not self.llm_final:
                logger.error("❌ Final LLM mevcut değil!")
                return "Yanıt oluşturma servisi geçici olarak kullanılamıyor. Lütfen tekrar deneyin."
            
            logger.info(f"🎯 FINAL RESPONSE OLUŞTURULUYOR:")
            logger.info(f"   📝 Soru: '{question[:60]}...'")
            logger.info(f"   📄 Context1: {len(context1)} karakter")
            logger.info(f"   📊 Context2: {len(context2)} karakter")
            logger.info(f"   🧠 History: {len(history)} karakter")
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2,
                    history=history
                )
            )
            
            final_response = result.content.strip()
            final_time = time.time() - final_start
            
            logger.info(f"✅ FINAL RESPONSE TAMAMLANDI ({final_time:.2f}s):")
            logger.info(f"   📝 Yanıt uzunluğu: {len(final_response)} karakter")
            logger.info(f"   📄 Yanıt önizleme: '{final_response[:100]}...'")
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Final yanıt hatası: {e}")
            return "Yanıt oluşturulurken hata oluştu. Lütfen tekrar deneyin."

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Kaynaklarını çıkar - Detaylı logging ile"""
        sources = []
        
        logger.info(f"🔍 KAYNAK ÇIKARIMI:")
        
        # Context1 (Vector) kaynak kontrolü
        if "Dosya:" in context1 and "bulunamadı" not in context1 and "başarısız" not in context1:
            sources.append(MessageSettings.SOURCES["YOK_REPORT"])
            sources.append(MessageSettings.SOURCES["IZU_GUIDE"])
            logger.info(f"   📄 Vector kaynakları eklendi: YÖK Raporu, İZÜ Rehberi")
        else:
            logger.info(f"   ❌ Vector kaynak bulunamadı")
        
        # Context2 (CSV) kaynak kontrolü
        if (context2 and 
            "mevcut değil" not in context2 and 
            "hata" not in context2 and 
            "başarısız" not in context2 and
            len(context2.strip()) > 50):  # Minimum content check
            sources.append(MessageSettings.SOURCES["UNIVERI_DB"])
            logger.info(f"   📊 CSV kaynağı eklendi: UNİ-VERİ DB")
        else:
            logger.info(f"   ❌ CSV kaynak bulunamadı")
        
        # Kaynak yoksa genel kaynak ekle
        if not sources:
            sources.append(MessageSettings.SOURCES["GENERAL"])
            logger.info(f"   📝 Genel kaynak eklendi")
        
        logger.info(f"   ✅ Toplam kaynak sayısı: {len(sources)}")
        
        return sources

    async def test_all_connections(self) -> Dict[str, str]:
        """Gelişmiş bağlantı testi - Detaylı logging"""
        logger.info("🧪 TÜM BAĞLANTILAR TEST EDİLİYOR...")
        results = {}
        
        # OpenAI Client test
        try:
            if self.openai_client:
                test_start = time.time()
                test_embedding = self.get_embedding("test")
                test_time = time.time() - test_start
                results["OpenAI Client"] = f"✅ Bağlı ({len(test_embedding)} boyut, {test_time:.2f}s)"
                logger.info(f"   ✅ OpenAI Client: OK")
            else:
                results["OpenAI Client"] = "❌ Client başlatılmadı"
                logger.error(f"   ❌ OpenAI Client: Başlatılmadı")
        except Exception as e:
            results["OpenAI Client"] = f"❌ Hata: {str(e)[:50]}"
            logger.error(f"   ❌ OpenAI Client: {e}")
        
        # LLM testleri
        llm_tests = [
            ("Evaluation", self.llm_evaluation),
            ("Correction", self.llm_correction), 
            ("Search Optimizer", self.llm_search_optimizer),
            ("CSV Agent", self.llm_csv_agent),
            ("Final Response", self.llm_final)
        ]
        
        for name, llm in llm_tests:
            try:
                if llm:
                    test_start = time.time()
                    await llm.ainvoke("Test")
                    test_time = time.time() - test_start
                    results[name] = f"✅ Bağlı ({test_time:.2f}s)"
                    logger.info(f"   ✅ {name}: OK")
                else:
                    results[name] = "❌ Model yüklenmedi"
                    logger.error(f"   ❌ {name}: Yüklenmedi")
            except Exception as e:
                results[name] = f"❌ Hata: {str(e)[:50]}"
                logger.error(f"   ❌ {name}: {e}")
        
        # Native AstraDB test
        try:
            if self.astra_collection:
                test_start = time.time()
                test_results = list(self.astra_collection.find({}, limit=1))
                test_time = time.time() - test_start
                results["AstraDB Native"] = f"✅ Bağlı ({len(test_results)} doküman, {test_time:.2f}s)"
                logger.info(f"   ✅ AstraDB Native: OK")
            else:
                results["AstraDB Native"] = "❌ Collection başlatılmadı"
                logger.error(f"   ❌ AstraDB Native: Başlatılmadı")
        except Exception as e:
            results["AstraDB Native"] = f"❌ Hata: {str(e)[:50]}"
            logger.error(f"   ❌ AstraDB Native: {e}")
        
        # CSV test
        try:
            if self.csv_data is not None:
                unique_bolumlr = len(self.csv_data['bolum_adi'].unique())
                results["CSV"] = f"✅ Yüklü ({len(self.csv_data)} satır, {unique_bolumlr} bölüm)"
                logger.info(f"   ✅ CSV: OK")
            else:
                results["CSV"] = "❌ Yüklenmedi"
                logger.error(f"   ❌ CSV: Yüklenmedi")
        except Exception as e:
            results["CSV"] = f"❌ Hata: {str(e)[:50]}"
            logger.error(f"   ❌ CSV: {e}")
        
        # Memory test
        try:
            test_start = time.time()
            self.memory.add_message("test_connection", "user", "test")
            history = self.memory.get_history("test_connection")
            test_time = time.time() - test_start
            
            if history:
                results["Memory"] = f"✅ Redis bağlı ({test_time:.3f}s)"
                logger.info(f"   ✅ Memory: OK")
            else:
                results["Memory"] = f"⚠️ Memory çalışıyor ama boş ({test_time:.3f}s)"
                logger.warning(f"   ⚠️ Memory: Çalışıyor ama boş")
        except Exception as e:
            results["Memory"] = f"❌ Hata: {str(e)[:50]}"
            logger.error(f"   ❌ Memory: {e}")
        
        # Test özeti
        success_count = sum(1 for v in results.values() if v.startswith("✅"))
        total_count = len(results)
        logger.info(f"🧪 TEST RAPORU: {success_count}/{total_count} başarılı")
        
        return results

    async def debug_csv_data(self) -> Dict[str, Any]:
        """CSV debug bilgileri - Detaylı analiz"""
        try:
            if self.csv_data is None:
                return {"status": "error", "message": "CSV verisi yüklenmedi"}
            
            logger.info("🧪 CSV DEBUG BİLGİLERİ TOPLANILIYOR...")
            
            debug_info = {
                "basic_info": {
                    "total_rows": len(self.csv_data),
                    "total_columns": len(self.csv_data.columns),
                    "unique_bolumlr": len(self.csv_data['bolum_adi'].unique()),
                    "unique_years": len(self.csv_data['gosterge_id'].unique()) if 'gosterge_id' in self.csv_data.columns else 0
                },
                "sample_bolumlr": self.csv_data['bolum_adi'].unique()[:10].tolist(),
                "column_categories": {
                    "istihdam": [col for col in self.csv_data.columns if "istihdam" in col],
                    "maas": [col for col in self.csv_data.columns if col.startswith("maas_")],
                    "firma": [col for col in self.csv_data.columns if col.startswith("firma_")],
                    "sektor": [col for col in self.csv_data.columns if col.startswith("sektor_")]
                },
                "sample_data": self.csv_data.head(3).to_dict(),
                "missing_data": self.csv_data.isnull().sum().to_dict()
            }
            
            logger.info(f"   📊 Toplam satır: {debug_info['basic_info']['total_rows']}")
            logger.info(f"   📋 Toplam kolon: {debug_info['basic_info']['total_columns']}")
            logger.info(f"   🎓 Toplam bölüm: {debug_info['basic_info']['unique_bolumlr']}")
            
            return {"status": "success", "debug_info": debug_info}
            
        except Exception as e:
            logger.error(f"❌ CSV debug hatası: {e}")
            return {"status": "error", "message": str(e)}
