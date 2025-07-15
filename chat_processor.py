import os
import pandas as pd
import re
from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
import math
import json
import asyncio
import re
from astrapy import DataAPIClient
from langchain_openai import OpenAIEmbeddings
from memory import ConversationMemory
from langchain_anthropic import ChatAnthropic
#from astrapy.info import VectorServiceOptions

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
        params = config.to_langchain_params()
        
        if config.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**params)
        elif config.provider == LLMProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(**params)

class TercihAsistaniProcessor:
    """
    Langflow akışınızı taklit eden ana processor - Config'lerle güncellenmiş
    """
    
    def __init__(self):
        self.llm_evaluation = None
        self.llm_correction = None
        self.llm_search_optimizer = None
        self.llm_csv_agent = None
        self.llm_final = None
        self.vectorstore = None
        self.csv_data = None
        self.memory = ConversationMemory() 
        
        # Config'lerden prompt'ları al
        self.evaluation_prompt = ChatPromptTemplate.from_template(PromptTemplates.EVALUATION)
        self.correction_prompt = ChatPromptTemplate.from_template(PromptTemplates.CORRECTION)
        self.search_optimizer_prompt = ChatPromptTemplate.from_template(PromptTemplates.SEARCH_OPTIMIZER)
        self.csv_agent_prompt = ChatPromptTemplate.from_template(PromptTemplates.CSV_AGENT)
        self.final_prompt = ChatPromptTemplate.from_template(PromptTemplates.FINAL_RESPONSE)

    async def initialize(self):
        """Tüm bileşenleri başlat - Config'lerle"""
  
        try:
            
            logger.info(f"🔑 Environment check:")
            logger.info(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
            logger.info(f"   ANTHROPIC_API_KEY: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")
            logger.info(f"   GOOGLE_API_KEY: {'✅ Set' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
            
            logger.info(f"   ANTHROPIC_API_KEY: {os.getenv('ANTHROPIC_API_KEY')}")
            # Multi-provider LLM'leri başlat
            self.llm_evaluation = LLMFactory.create_llm(LLMConfigs.EVALUATION)
            self.llm_correction = LLMFactory.create_llm(LLMConfigs.CORRECTION)
            self.llm_search_optimizer = LLMFactory.create_llm(LLMConfigs.SEARCH_OPTIMIZER)
            self.llm_csv_agent = LLMFactory.create_llm(LLMConfigs.CSV_AGENT)
            self.llm_final = LLMFactory.create_llm(LLMConfigs.FINAL_RESPONSE)
            
            logger.info("LLM modelleri başlatıldı")
            
            # AstraDB bağlantısı
            await self._initialize_astradb()
            
            # CSV verilerini yükle
            await self._initialize_csv()
                
            logger.info("TercihAsistaniProcessor başlatıldı")
            
        except Exception as e:
            logger.error(f"Initialization hatası: {e}")
            raise

    async def _initialize_astradb(self):
        """AstraDB bağlantısını başlat - Ultra minimal"""
        try:
            # ASYNC client
            client = DataAPIClient(token=DatabaseSettings.ASTRA_DB_TOKEN)
    
            # ASYNC database handle al
            async_database = client.get_async_database(DatabaseSettings.ASTRA_DB_API_ENDPOINT)
    
            # Koleksiyonları ASYNC listele
            collections = await async_database.list_collection_names()
    
            logger.info(f"Mevcut collection'lar: {collections}")
    
            collection_name = DatabaseSettings.ASTRA_DB_COLLECTION
    
            if collection_name in collections:
                logger.info(f"Collection bulundu: {collection_name}")
    
                try:
                    embedding = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        dimensions=512,  # 512 boyuta ayarla
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )

                    self.vectorstore = AstraDBVectorStore(
                        token=DatabaseSettings.ASTRA_DB_TOKEN,
                        api_endpoint=DatabaseSettings.ASTRA_DB_API_ENDPOINT,
                        collection_name=collection_name,
                        embedding=embedding,
                        # collection_vector_service_options parametresini kaldır
                    )
    
                    logger.info("✅ AstraDB VectorStore başarıyla oluşturuldu!")
    
                    # Vector aramayı da ASYNC yapmak istersen
                    test_docs = self.vectorstore.similarity_search("test", k=1)
                    logger.info(f"✅ Test arama başarılı: {len(test_docs)} doküman bulundu")
    
                except Exception as vs_error:
                    logger.error(f"❌ AstraDBVectorStore oluşturma hatası: {vs_error}")
                    self.vectorstore = None
    
            else:
                logger.error(f"Collection '{collection_name}' bulunamadı!")
                self.vectorstore = None
    
        except Exception as e:
            logger.error(f"AstraDB genel bağlantı hatası: {e}")
            self.vectorstore = None
    
    async def _initialize_csv(self):
        """CSV verilerini yükle"""
        try:
            csv_path = DatabaseSettings.CSV_FILE_PATH
            if csv_path and os.path.exists(csv_path):
                self.csv_data = pd.read_csv(csv_path)
                logger.info(f"CSV verisi yüklendi: {len(self.csv_data)} satır")
            else:
                logger.warning(f"CSV dosyası bulunamadı: {csv_path}")
                self.csv_data = None
        except Exception as e:
            logger.warning(f"CSV yükleme hatası: {e}")
            self.csv_data = None

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Ana mesaj işleme akışı - Langflow'daki akışınızın aynısı
        """
        try:
            # chat_processor.py'de debug ekle:
            logger.info(f"🔍 RAW Session ID: '{session_id}' - Type: {type(session_id)} - Length: {len(session_id)}")
            
            # Adım 1: Soru uygunluk değerlendirmesi
            evaluation_result = await self._evaluate_question(message)
            
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
            
            # Adım 3: Soru düzeltme (sadece uygun sorular için)
            corrected_question = await self._correct_question(message)
            
            # Adım 4: Paralel işlemler
            context1, context2 = await asyncio.gather(
                self._get_vector_context(corrected_question),
                self._get_csv_context(corrected_question)
            )
            
            # Adım 5: Final yanıt oluşturma
            # DEBUG LOG'LAR EKLE:
            logger.info(f"🔍 Session ID: {session_id}")
            conversation_history = self.memory.get_history(session_id)
            logger.info(f"🧠 Memory history length: {len(conversation_history)} - Content: '{conversation_history[:100]}...'")
           
            final_response = await self._generate_final_response(
                question=corrected_question,
                context1=context1,
                context2=context2,
                history=conversation_history
            )

            # DEBUG LOG EKLE:
            logger.info(f"💾 Memory'ye kaydediliyor - Session: {session_id}")
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", final_response)
            logger.info(f"📝 Memory kayıt tamamlandı")

            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2)
            }
            
        except Exception as e:
            logger.error(f"Mesaj işleme hatası: {e}")
            return {
                "response": MessageSettings.ERROR_GENERAL,
                "sources": []
            }

    async def _evaluate_question(self, question: str) -> str:
        try:
            result = await self.llm_evaluation.ainvoke(
                self.evaluation_prompt.format(question=question)
            )
            evaluation_result = result.content.strip()
            logger.info(f"Soru değerlendirme sonucu: {evaluation_result}")
            
            # SELAMLAMA kontrolü ekle
            if "SELAMLAMA" in evaluation_result.upper():
                return "SELAMLAMA"
            elif "UYGUN" in evaluation_result.upper():
                return "UYGUN"
            else:
                return "Uzmanlık dışı soru"
                
        except Exception as e:
            logger.error(f"Değerlendirme hatası: {e}")
            return "UYGUN"

    async def _correct_question(self, question: str) -> str:
        """Soru düzeltme ve standardizasyon"""
        try:
            result = await self.llm_correction.ainvoke(
                self.correction_prompt.format(question=question)
            )
            corrected = result.content.strip()
            logger.info(f"Düzeltilmiş soru: {corrected}")
            return corrected
        except Exception as e:
            logger.error(f"Düzeltme hatası: {e}")
            return question

    async def _get_vector_context(self, question: str) -> str:
        """Vector database'den context al"""
        try:
            if not self.vectorstore:
                logger.warning("Vector store mevcut değil")
                return "Vector arama mevcut değil"
            
            logger.info(f"Vector arama yapılıyor: {question}")
            
            # Arama sorgusunu optimize et
            try:
                optimized_query = await self.llm_search_optimizer.ainvoke(
                    self.search_optimizer_prompt.format(question=question)
                )
                optimized_text = optimized_query.content.strip()
                logger.info(f"Optimize edilmiş sorgu: {optimized_text}")
            except Exception as e:
                logger.warning(f"Sorgu optimizasyonu başarısız, orijinal soru kullanılıyor: {e}")
                optimized_text = question
            
            # Vector arama yap
            try:
                docs = self.vectorstore.similarity_search(
                    optimized_text, 
                    k=VectorConfig.SIMILARITY_TOP_K
                )
                logger.info(f"Bulunan doküman sayısı: {len(docs)}")
                
                if not docs:
                    logger.warning("Hiç doküman bulunamadı")
                    return "İlgili doküman bulunamadı"
                
                # Dokümanları birleştir
                context = ""
                for i, doc in enumerate(docs):
                    # Metadata'dan dosya yolu al
                    file_path = doc.metadata.get('file_path', 'Bilinmeyen kaynak')
                    content = doc.page_content[:500]  # İlk 500 karakter
                    context += f"Dosya: {file_path}\nİçerik: {content}\n\n"
                    logger.info(f"Doküman {i+1}: {file_path} - İçerik uzunluğu: {len(doc.page_content)}")
                
                logger.info(f"Toplam context uzunluğu: {len(context)}")
                return context
                
            except Exception as e:
                logger.error(f"Vector arama hatası: {e}")
                return f"Vector arama sırasında hata oluştu: {str(e)}"
            
        except Exception as e:
            logger.error(f"Vector context genel hatası: {e}")
            return f"Vector arama sırasında genel hata oluştu: {str(e)}"



 

    async def _get_csv_context(self, question: str) -> str:
        """Akıllı CSV analiz - bölüm, yıl ve metrik filtreleme"""
        try:
            if self.csv_data is None:
                return "CSV verileri mevcut değil"
    
            question_lower = question.lower()
    
            # Bölüm adını bul
            bolum_adi = None
            for bolum in self.csv_data['bolum_adi'].unique():
                if bolum.lower() in question_lower:
                    bolum_adi = bolum
                    break
    
            # Gösterge ID / yıl
            gosterge_id = None
            match = re.search(r'\b20\d{2}\b', question)
            if match:
                gosterge_id = int(match.group(0))
    
            # Metrik çıkarımı
            metrik_map = {
                "istihdam": [
                    "istihdam_orani",
                    "akademik_istihdam_orani",
                    "yonetici_pozisyonu_istihdam_orani"
                ],
                "maaş": [col for col in self.csv_data.columns if col.startswith("maas_")],
                "firma": [col for col in self.csv_data.columns if col.startswith("firma_")],
                "girişim": ["girisimcilik_orani", "katma_degerli_girisim_endeksi"] + 
                           [col for col in self.csv_data.columns if col.startswith("girisim_omru_")],
                "sektör": [col for col in self.csv_data.columns if col.startswith("sektor_")]
            }
    
            metrikler = []
            for anahtar, cols in metrik_map.items():
                if anahtar in question_lower:
                    metrikler.extend(cols)
    
            if not metrikler:
                # Hiç anahtar eşleşmezse default: tüm metrikler
                metrikler = [col for col in self.csv_data.columns if col not in ['bolum_adi', 'gosterge_id', 'bolum_id']]
    
            # Filtre oluştur
            criteria = []
            if bolum_adi:
                criteria.append(self.csv_data['bolum_adi'] == bolum_adi)
            if gosterge_id:
                criteria.append(self.csv_data['gosterge_id'] == gosterge_id)
    
            if criteria:
                mask = criteria[0]
                for cond in criteria[1:]:
                    mask &= cond
                filtered = self.csv_data[mask]
            else:
                filtered = self.csv_data
    
            if filtered.empty:
                logger.warning("Filtre sonucunda veri bulunamadı. Örnek veri gönderiliyor.")
                filtered = self.csv_data.head(CSVConfig.SAMPLE_ROWS)
    
            selected_cols = ['bolum_adi', 'gosterge_id'] + metrikler
            selected = filtered[selected_cols]
    
            logger.info(f"Filtreli CSV veri: {selected.shape} - Bölüm: {bolum_adi} - Yıl: {gosterge_id}")
    
            csv_snippet = selected.to_string(index=False)
    
            result = await self.llm_csv_agent.ainvoke(
                self.csv_agent_prompt.format(
                    question=question,
                    csv_data=csv_snippet
                )
            )
    
            analysis = result.content.strip()
            logger.info(f"Akıllı CSV analiz sonucu: {analysis[:200]}...")
            return analysis
    
        except Exception as e:
            logger.error(f"Akıllı CSV analiz hatası: {e}")
            return "CSV analizi sırasında hata oluştu"



    async def _generate_final_response(self, question: str, context1: str, context2: str , history: str = "") -> str:
        """Final yanıtı oluştur"""
        try:
            logger.info(f"Final yanıt oluşturuluyor - Context1: {len(context1)}, Context2: {len(context2)}")
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2,
                    history=history
                )
            )
            
            final_response = result.content.strip()
            logger.info(f"Final yanıt oluşturuldu: {len(final_response)} karakter")
            return final_response
            
        except Exception as e:
            logger.error(f"Final yanıt hatası: {e}")
            return "Yanıt oluşturulurken hata oluştu"

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Kaynaklarını çıkar - Config'lerden"""
        sources = []
        
        if "Dosya:" in context1:
            sources.append(MessageSettings.SOURCES["YOK_REPORT"])
            sources.append(MessageSettings.SOURCES["IZU_GUIDE"])
        
        if context2 and "CSV" not in context2 and "gerekli değil" not in context2:
            sources.append(MessageSettings.SOURCES["UNIVERI_DB"])
        
        return sources

    async def test_all_connections(self) -> Dict[str, str]:
        """Tüm bağlantıları test et"""
        results = {}
        
        # OpenAI test
        try:
            test_result = await self.llm_evaluation.ainvoke("Test")
            results["OpenAI"] = "✅ Bağlı"
        except Exception as e:
            results["OpenAI"] = f"❌ Bağlantı hatası: {str(e)[:50]}"
        
        # AstraDB test
        try:
            if self.vectorstore:
                self.vectorstore.similarity_search("test", k=1)
                results["AstraDB"] = "✅ Bağlı"
            else:
                results["AstraDB"] = "❌ Yapılandırılmamış"
        except Exception as e:
            results["AstraDB"] = f"❌ Bağlantı hatası: {str(e)[:50]}"
        
        # CSV test
        if self.csv_data is not None:
            results["CSV"] = f"✅ Yüklü ({len(self.csv_data)} satır)"
        else:
            results["CSV"] = "❌ Yüklenmedi"
        
        return results

