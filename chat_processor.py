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
    Astrapy native API ile güncellenmiş processor
    """
    
    def __init__(self):
        self.llm_evaluation = None
        self.llm_correction = None
        self.llm_search_optimizer = None
        self.llm_csv_agent = None
        self.llm_final = None
        
        # YENİ: Astrapy native components
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
        """Gelişmiş hata yönetimi ile mesaj işleme"""
        try:
            logger.info(f"📨 Mesaj işleniyor - Session: {session_id}")
            
            # Adım 1: Soru uygunluk değerlendirmesi (hata toleranslı)
            evaluation_result = await self._evaluate_question_safe(message)
            
            # Adım 2: Koşullu yönlendirme
            if evaluation_result == "Uzmanlık dışı soru":
                return {
                    "response": MessageSettings.ERROR_EXPERTISE_OUT,
                    "sources": []
                }
            
            if evaluation_result == "SELAMLAMA":
                return {
                    "response": "Merhaba! Ben bir üniversite tercih asistanıyım. Size YKS tercihleri, bölüm seçimi, kariyer planlaması konularında yardımcı olabilirim. Hangi konuda bilgi almak istiyorsunuz?",
                    "sources": []
                }
            
            # Adım 3: Soru düzeltme (güvenli)
            corrected_question = await self._correct_question_safe(message)
            
            # Adım 4: Paralel işlemler (hata toleranslı)
            context1, context2 = await asyncio.gather(
                self._get_vector_context_native(corrected_question),
                self._get_csv_context_safe(corrected_question),
                return_exceptions=True
            )
            
            # Exception'ları handle et
            if isinstance(context1, Exception):
                logger.error(f"Vector context hatası: {context1}")
                context1 = "Vector arama başarısız"
                
            if isinstance(context2, Exception):
                logger.error(f"CSV context hatası: {context2}")
                context2 = "CSV analizi başarısız"
            
            # Adım 5: Final yanıt oluşturma
            conversation_history = self.memory.get_history(session_id)
            final_response = await self._generate_final_response_safe(
                question=corrected_question,
                context1=context1,
                context2=context2,
                history=conversation_history
            )

            # Memory'ye kaydet
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", final_response)

            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2)
            }
            
        except Exception as e:
            logger.error(f"❌ Mesaj işleme hatası: {e}")
            return {
                "response": MessageSettings.ERROR_GENERAL,
                "sources": []
            }

    async def _evaluate_question_safe(self, question: str) -> str:
        """Güvenli soru değerlendirme"""
        try:
            if not self.llm_evaluation:
                logger.warning("Evaluation LLM mevcut değil, varsayılan UYGUN")
                return "UYGUN"
                
            result = await self.llm_evaluation.ainvoke(
                self.evaluation_prompt.format(question=question)
            )
            evaluation_result = result.content.strip()
            logger.info(f"✅ Soru değerlendirme: {evaluation_result}")
            
            if "SELAMLAMA" in evaluation_result.upper():
                return "SELAMLAMA"
            elif "UYGUN" in evaluation_result.upper():
                return "UYGUN"
            else:
                return "Uzmanlık dışı soru"
                
        except Exception as e:
            logger.error(f"❌ Değerlendirme hatası: {e}")
            return "UYGUN"  # Güvenli varsayılan

    async def _correct_question_safe(self, question: str) -> str:
        """Güvenli soru düzeltme"""
        try:
            if not self.llm_correction:
                logger.warning("Correction LLM mevcut değil, orijinal soru")
                return question
                
            result = await self.llm_correction.ainvoke(
                self.correction_prompt.format(question=question)
            )
            corrected = result.content.strip()
            logger.info(f"✅ Düzeltilmiş soru: {corrected}")
            return corrected
        except Exception as e:
            logger.error(f"❌ Düzeltme hatası: {e}")
            return question

    async def _get_vector_context_native(self, question: str) -> str:
        """Native Astrapy API ile vector search"""
        try:
            if not self.astra_collection:
                logger.warning("Astra collection mevcut değil")
                return "Vector arama mevcut değil"
                
            # Query optimization (güvenli)
            optimized_text = question
            if self.llm_search_optimizer:
                try:
                    optimized_query = await self.llm_search_optimizer.ainvoke(
                        self.search_optimizer_prompt.format(question=question)
                    )
                    optimized_text = optimized_query.content.strip()
                    logger.info(f"✅ Optimize edilmiş sorgu: {optimized_text}")
                except Exception as e:
                    logger.warning(f"Query optimization hatası: {e}")
            
            # Embedding oluştur
            query_embedding = self.get_embedding(optimized_text)
            logger.info(f"✅ Query embedding oluşturuldu: {len(query_embedding)} boyut")
            
            # Native vector search
            results = self.astra_collection.find(
                {},
                sort={"$vector": query_embedding},
                limit=VectorConfig.SIMILARITY_TOP_K
            )
            
            # Sonuçları işle
            docs = list(results)
            logger.info(f"✅ Vector search sonuç: {len(docs)} doküman bulundu")
            
            if not docs:
                return "İlgili doküman bulunamadı"
            
            # Context oluştur
            context = ""
            for i, doc in enumerate(docs):
                # Astrapy'da metadata farklı yapıda olabilir
                file_path = doc.get('metadata', {}).get('file_path', doc.get('file_path', 'Bilinmeyen kaynak'))
                content = doc.get('content', doc.get('text', str(doc)))[:500]
                context += f"Dosya: {file_path}\nİçerik: {content}\n\n"
            
            logger.info(f"✅ Native vector context: {len(context)} karakter")
            return context
            
        except Exception as e:
            logger.error(f"❌ Native vector context hatası: {e}")
            return f"Vector arama hatası: {str(e)[:100]}"

    async def _get_csv_context_safe(self, question: str) -> str:
        """Güvenli CSV analiz"""
        try:
            if self.csv_data is None:
                return "CSV verileri mevcut değil"
            
            if not self.llm_csv_agent:
                logger.error("❌ CSV Agent LLM mevcut değil!")
                return "CSV analizi için gerekli model yüklenmedi"
            
            # CSV filtreleme (aynı logic)
            question_lower = question.lower()
            
            # Bölüm adını bul
            bolum_adi = None
            for bolum in self.csv_data['bolum_adi'].unique():
                if bolum.lower() in question_lower:
                    bolum_adi = bolum
                    break
            
            # Filtering logic (aynı)
            metrik_map = {
                "istihdam": [col for col in self.csv_data.columns if "istihdam" in col],
                "maaş": [col for col in self.csv_data.columns if col.startswith("maas_")],
                "firma": [col for col in self.csv_data.columns if col.startswith("firma_")],
                "girişim": [col for col in self.csv_data.columns if "girisim" in col],
                "sektör": [col for col in self.csv_data.columns if col.startswith("sektor_")]
            }
            
            metrikler = []
            for anahtar, cols in metrik_map.items():
                if anahtar in question_lower:
                    metrikler.extend(cols)
            
            if not metrikler:
                metrikler = [col for col in self.csv_data.columns if col not in ['bolum_adi', 'gosterge_id', 'bolum_id']]
            
            # Filter uygula
            filtered = self.csv_data
            if bolum_adi:
                filtered = filtered[filtered['bolum_adi'] == bolum_adi]
            
            if filtered.empty:
                filtered = self.csv_data.head(CSVConfig.SAMPLE_ROWS)
            
            selected_cols = ['bolum_adi', 'gosterge_id'] + metrikler
            selected = filtered[selected_cols]
            
            csv_snippet = selected.to_string(index=False)
            
            # CSV Agent çağrısı
            result = await self.llm_csv_agent.ainvoke(
                self.csv_agent_prompt.format(
                    question=question,
                    csv_data=csv_snippet
                )
            )
            
            analysis = result.content.strip()
            logger.info(f"✅ CSV analiz: {len(analysis)} karakter")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ CSV analiz hatası: {e}")
            return "CSV analizi sırasında hata oluştu"

    async def _generate_final_response_safe(self, question: str, context1: str, context2: str, history: str = "") -> str:
        """Güvenli final yanıt oluşturma"""
        try:
            if not self.llm_final:
                return "Yanıt oluşturma servisi geçici olarak kullanılamıyor. Lütfen tekrar deneyin."
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2,
                    history=history
                )
            )
            
            final_response = result.content.strip()
            logger.info(f"✅ Final yanıt: {len(final_response)} karakter")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Final yanıt hatası: {e}")
            return "Yanıt oluşturulurken hata oluştu. Lütfen tekrar deneyin."

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Kaynaklarını çıkar"""
        sources = []
        
        if "Dosya:" in context1 and "bulunamadı" not in context1:
            sources.append(MessageSettings.SOURCES["YOK_REPORT"])
            sources.append(MessageSettings.SOURCES["IZU_GUIDE"])
        
        if context2 and "mevcut değil" not in context2 and "hata" not in context2:
            sources.append(MessageSettings.SOURCES["UNIVERI_DB"])
        
        return sources

    async def test_all_connections(self) -> Dict[str, str]:
        """Gelişmiş bağlantı testi"""
        results = {}
        
        # OpenAI Client test
        try:
            if self.openai_client:
                test_embedding = self.get_embedding("test")
                results["OpenAI Client"] = f"✅ Bağlı ({len(test_embedding)} boyut embedding)"
            else:
                results["OpenAI Client"] = "❌ Client başlatılmadı"
        except Exception as e:
            results["OpenAI Client"] = f"❌ Hata: {str(e)[:50]}"
        
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
                    await llm.ainvoke("Test")
                    results[name] = "✅ Bağlı ve çalışıyor"
                else:
                    results[name] = "❌ Model yüklenmedi"
            except Exception as e:
                results[name] = f"❌ Hata: {str(e)[:50]}"
        
        # Native AstraDB test
        try:
            if self.astra_collection:
                test_results = list(self.astra_collection.find({}, limit=1))
                results["AstraDB Native"] = f"✅ Bağlı ({len(test_results)} test doküman)"
            else:
                results["AstraDB Native"] = "❌ Collection başlatılmadı"
        except Exception as e:
            results["AstraDB Native"] = f"❌ Hata: {str(e)[:50]}"
        
        # CSV test
        if self.csv_data is not None:
            results["CSV"] = f"✅ Yüklü ({len(self.csv_data)} satır)"
        else:
            results["CSV"] = "❌ Yüklenmedi"
        
        # Memory test
        try:
            self.memory.add_message("test", "user", "test")
            history = self.memory.get_history("test")
            results["Memory"] = "✅ Redis bağlı" if history else "⚠️ Memory çalışıyor ama boş"
        except Exception as e:
            results["Memory"] = f"❌ Hata: {str(e)[:50]}"
        
        return results
