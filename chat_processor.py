import os
import pandas as pd
import re
from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Config imports
from config import (
    LLMConfigs, VectorConfig, CSVConfig,
    PromptTemplates, CSV_KEYWORDS,
    DatabaseSettings, MessageSettings
)

logger = logging.getLogger(__name__)

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
        
        # Config'lerden prompt'ları al
        self.evaluation_prompt = ChatPromptTemplate.from_template(PromptTemplates.EVALUATION)
        self.correction_prompt = ChatPromptTemplate.from_template(PromptTemplates.CORRECTION)
        self.search_optimizer_prompt = ChatPromptTemplate.from_template(PromptTemplates.SEARCH_OPTIMIZER)
        self.csv_agent_prompt = ChatPromptTemplate.from_template(PromptTemplates.CSV_AGENT)
        self.final_prompt = ChatPromptTemplate.from_template(PromptTemplates.FINAL_RESPONSE)

    async def initialize(self):
        """Tüm bileşenleri başlat - Config'lerle"""
        try:
            # OpenAI modellerini config'lerden başlat
            self.llm_evaluation = ChatOpenAI(**LLMConfigs.EVALUATION.to_dict())
            self.llm_correction = ChatOpenAI(**LLMConfigs.CORRECTION.to_dict())
            self.llm_search_optimizer = ChatOpenAI(**LLMConfigs.SEARCH_OPTIMIZER.to_dict())
            self.llm_csv_agent = ChatOpenAI(**LLMConfigs.CSV_AGENT.to_dict())
            self.llm_final = ChatOpenAI(**LLMConfigs.FINAL_RESPONSE.to_dict())
            
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
        """AstraDB bağlantısını başlat"""
    try:
        from astrapy import DataAPIClient
        
        # Collection'ları listele (debug için)
        client = DataAPIClient(token=DatabaseSettings.ASTRA_DB_TOKEN)
        database = client.get_database_by_api_endpoint(DatabaseSettings.ASTRA_DB_API_ENDPOINT)
        collections = list(database.list_collection_names())
        logger.info(f"Mevcut collection'lar: {collections}")
        
        collection_name = DatabaseSettings.ASTRA_DB_COLLECTION
        if collection_name in collections:
            logger.info(f"Mevcut collection kullanılıyor: {collection_name}")
            
            # Minimal AstraDBVectorStore
            self.vectorstore = AstraDBVectorStore(
                token=DatabaseSettings.ASTRA_DB_TOKEN,
                api_endpoint=DatabaseSettings.ASTRA_DB_API_ENDPOINT,
                collection_name=collection_name,
                embedding=None
            )
            logger.info("AstraDB bağlantısı başarılı!")
        else:
            logger.error(f"Collection '{collection_name}' bulunamadı!")
            self.vectorstore = None
            
    except Exception as e:
        logger.error(f"AstraDB bağlantı hatası: {e}")
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
            # Adım 1: Soru uygunluk değerlendirmesi
            evaluation_result = await self._evaluate_question(message)
            
            # Adım 2: Koşullu yönlendirme
            if evaluation_result.strip() == "Uzmanlık dışı soru":
                return {
                    "response": MessageSettings.ERROR_EXPERTISE_OUT,
                    "sources": []
                }
            
            # Adım 3: Soru düzeltme
            corrected_question = await self._correct_question(message)
            
            # Adım 4: Paralel işlemler
            # 4a: Vector arama
            context1 = await self._get_vector_context(corrected_question)
            
            # 4b: CSV analizi
            context2 = await self._get_csv_context(corrected_question)
            
            # Adım 5: Final yanıt oluşturma
            final_response = await self._generate_final_response(
                question=corrected_question,
                context1=context1,
                context2=context2
            )
            
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
        """Soru uygunluk değerlendirmesi"""
        try:
            result = await self.llm_evaluation.ainvoke(
                self.evaluation_prompt.format(question=question)
            )
            logger.info(f"Soru değerlendirme sonucu: {result.content.strip()}")
            return result.content.strip()
        except Exception as e:
            logger.error(f"Değerlendirme hatası: {e}")
            return question  # Hata durumunda soruyu olduğu gibi geçir

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
        """CSV verilerinden context al - Config'lerle"""
        try:
            if self.csv_data is None:
                return "CSV verileri mevcut değil"
            
            # Config'ten anahtar kelimeleri al
            question_lower = question.lower()
            has_csv_keyword = any(keyword.lower() in question_lower for keyword in CSV_KEYWORDS)
            
            if not has_csv_keyword:
                # Ek kontrol: sayısal değerler (maaş, yıl, oran için)
                has_numbers = bool(re.search(r'\d+', question))
                if not has_numbers:
                    return "Bu soru için CSV verisi gerekli değil"
            
            logger.info(f"CSV analizi gerekli: {question}")
            
            # CSV agent prompt ile analiz et - Config'e göre
            if len(self.csv_data) <= CSVConfig.MAX_ROWS_FOR_FULL_ANALYSIS:
                csv_sample = self.csv_data.to_string()
            else:
                csv_sample = self.csv_data.head(CSVConfig.SAMPLE_ROWS).to_string()
            
            result = await self.llm_csv_agent.ainvoke(
                self.csv_agent_prompt.format(
                    question=question,
                    csv_data=csv_sample
                )
            )
            
            analysis = result.content.strip()
            logger.info(f"CSV analiz sonucu: {analysis[:200]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"CSV analiz hatası: {e}")
            return "CSV analizi sırasında hata oluştu"

    async def _generate_final_response(self, question: str, context1: str, context2: str) -> str:
        """Final yanıtı oluştur"""
        try:
            logger.info(f"Final yanıt oluşturuluyor - Context1: {len(context1)}, Context2: {len(context2)}")
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2
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

