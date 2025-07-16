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
    DatabaseSettings, MessageSettings, AppSettings
)

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(config):
        """Gelişmiş hata yönetimi ile LLM oluştur"""
        try:
            params = config.to_langchain_params()
            if AppSettings.DEBUG_MODE:
                logger.debug(f"LLM creating: {config.provider.value} - {config.model}")
            
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
                raise ValueError(f"Unsupported provider: {config.provider}")
                
        except Exception as e:
            logger.error(f"LLM creation error ({config.provider.value}): {e}")
            raise

class TercihAsistaniProcessor:
    """
    Güncellenmiş processor - Smart Evaluator-Corrector ile
    """
    
    def __init__(self):
        # YENİ: Smart Evaluator-Corrector
        self.llm_smart_evaluator_corrector = None
        
        # KALAN LLM'LER
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
        
        # KALAN PROMPT'LAR
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
            logger.error(f"Embedding creation error: {e}")
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
            
            if AppSettings.LOG_MEMORY_OPERATIONS:
                logger.debug(f"Recent history retrieved: {len(recent_history)} chars")
            
            return recent_history
            
        except Exception as e:
            logger.error(f"Recent history error: {e}")
            return ""

    async def initialize(self):
        """Güncellenmiş başlatma - Smart Evaluator ile"""
        try:
            logger.info("TercihAsistaniProcessor initializing...")
            
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
                
            logger.info("TercihAsistaniProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
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
        
        logger.info("API Key status check:")
        for key, value in keys.items():
            status = "Set" if value else "Missing"
            if AppSettings.DEBUG_MODE:
                logger.debug(f"   {key}: {status}")
            
        # Critical keys check
        if not keys["OPENAI_API_KEY"]:
            raise ValueError("OPENAI_API_KEY is required!")

    def _initialize_openai_client(self):
        """OpenAI client'ı başlat"""
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            raise

    async def _initialize_llms_new(self):
        """YENİ LLM başlatma"""
        llm_configs = {
            "smart_evaluator_corrector": LLMConfigs.SMART_EVALUATOR_CORRECTOR,
            "csv_agent": LLMConfigs.CSV_AGENT,
            "final": LLMConfigs.FINAL_RESPONSE
        }
        
        for name, config in llm_configs.items():
            try:
                llm = LLMFactory.create_llm(config)
                setattr(self, f"llm_{name}", llm)
                logger.info(f"{name} LLM initialized: {config.model}")
                
            except Exception as e:
                logger.error(f"{name} LLM error: {e}")
                
                # Critical LLM'ler için fallback
                if name in ["smart_evaluator_corrector", "csv_agent", "final"]:
                    logger.warning(f"{name} using OpenAI fallback...")
                    fallback_config = LLMConfigs.FINAL_RESPONSE  # OpenAI model
                    try:
                        llm = LLMFactory.create_llm(fallback_config)
                        setattr(self, f"llm_{name}", llm)
                        logger.info(f"{name} fallback successful")
                    except Exception as fb_error:
                        logger.error(f"{name} fallback failed: {fb_error}")
                        setattr(self, f"llm_{name}", None)

    async def _initialize_astradb_native(self):
        """AstraDB native API ile bağlantı"""
        try:
            logger.info("Initializing AstraDB native API connection...")
            
            # Astra client oluştur
            astra_client = DataAPIClient(DatabaseSettings.ASTRA_DB_TOKEN)
            
            # Database bağlantısı
            self.astra_database = astra_client.get_database(
                DatabaseSettings.ASTRA_DB_API_ENDPOINT
            )
            
            # Collection al
            collection_name = DatabaseSettings.ASTRA_DB_COLLECTION
            self.astra_collection = self.astra_database.get_collection(collection_name)
            
            logger.info(f"AstraDB connected - Collection: {collection_name}")
            
            # Test sorgusu
            test_results = list(self.astra_collection.find({}, limit=1))
            logger.info(f"Connection test successful: {len(test_results)} documents found")
            
        except Exception as e:
            logger.error(f"AstraDB connection error: {e}")
            self.astra_database = None
            self.astra_collection = None

    async def _initialize_csv(self):
        """CSV verilerini güvenli yükle"""
        try:
            csv_path = DatabaseSettings.CSV_FILE_PATH
            
            if not csv_path or not os.path.exists(csv_path):
                logger.warning(f"CSV file not found: {csv_path}")
                self.csv_data = None
                return
            
            self.csv_data = pd.read_csv(csv_path)
            
            # Veri validasyonu
            if self.csv_data.empty:
                logger.warning("CSV file is empty")
                self.csv_data = None
                return
            
            # Gerekli kolonların varlığını kontrol et
            required_cols = ['bolum_adi', 'gosterge_id']
            missing_cols = [col for col in required_cols if col not in self.csv_data.columns]
            
            if missing_cols:
                logger.error(f"Missing CSV columns: {missing_cols}")
                self.csv_data = None
                return
            
            logger.info(f"CSV data loaded: {len(self.csv_data)} rows, {len(self.csv_data.columns)} columns")
            
        except Exception as e:
            logger.error(f"CSV loading error: {e}")
            self.csv_data = None

    async def _smart_evaluate_and_correct(self, message: str, session_id: str) -> Dict[str, str]:
        """YENİ: Smart Evaluator-Corrector fonksiyonu"""
        try:
            smart_start = time.time()
            
            if not self.llm_smart_evaluator_corrector:
                if AppSettings.LOG_LLM_RESPONSES:
                    logger.debug("Smart Evaluator-Corrector LLM unavailable, fallback")
                return {
                    "status": "UYGUN",
                    "enhanced_question": message
                }
            
            # Son birkaç mesajı al
            recent_history = self._get_recent_history(session_id, limit=4)
            
            if AppSettings.LOG_LLM_RESPONSES:
                logger.debug(f"Smart Evaluator starting: message_len={len(message)}, history_len={len(recent_history)}")
            
            # Smart Evaluator-Corrector'a gönder
            result = await self.llm_smart_evaluator_corrector.ainvoke(
                self.smart_evaluator_corrector_prompt.format(
                    question=message,
                    history=recent_history
                )
            )
            
            response = result.content.strip()
            smart_time = time.time() - smart_start
            
            if AppSettings.LOG_LLM_RESPONSES:
                logger.debug(f"Smart Evaluator response ({smart_time:.2f}s): {response[:100]}...")
            
            # Response'u parse et
            try:
                # STATUS ve ENHANCED_QUESTION'u extract et
                status_match = re.search(r'STATUS:\s*(\w+)', response)
                question_match = re.search(r'ENHANCED_QUESTION:\s*(.+)', response, re.DOTALL)
                
                if status_match and question_match:
                    status = status_match.group(1).strip()
                    enhanced_question = question_match.group(1).strip()
                    
                    if AppSettings.LOG_LLM_RESPONSES:
                        logger.debug(f"Parse successful: Status={status}, Enhanced_len={len(enhanced_question)}")
                    
                    return {
                        "status": status,
                        "enhanced_question": enhanced_question
                    }
                else:
                    if AppSettings.LOG_LLM_RESPONSES:
                        logger.debug("Parse failed - format error, using fallback")
                    
                    # Fallback parsing
                    if "UYGUN" in response.upper():
                        return {"status": "UYGUN", "enhanced_question": message}
                    elif "SELAMLAMA" in response.upper():
                        return {"status": "SELAMLAMA", "enhanced_question": message}
                    else:
                        return {"status": "KAPSAM_DIŞI", "enhanced_question": message}
                        
            except Exception as parse_error:
                logger.error(f"Smart Evaluator parse error: {parse_error}")
                return {"status": "UYGUN", "enhanced_question": message}
            
        except Exception as e:
            smart_time = time.time() - smart_start
            logger.error(f"Smart Evaluator-Corrector error ({smart_time:.2f}s): {e}")
            return {"status": "UYGUN", "enhanced_question": message}

    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """YENİ akış ile mesaj işleme"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing message: session={session_id}, length={len(message)}")
            
            # Adım 1: YENİ Smart Evaluator-Corrector
            smart_start = time.time()
            smart_result = await self._smart_evaluate_and_correct(message, session_id)
            smart_time = time.time() - smart_start
            
            status = smart_result["status"]
            enhanced_question = smart_result["enhanced_question"]
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"Smart Evaluator-Corrector: {smart_time:.2f}s, Status: {status}")
            
            # Adım 2: Koşullu yönlendirme
            if status == "KAPSAM_DIŞI":
                total_time = time.time() - start_time
                if AppSettings.PERFORMANCE_LOGGING:
                    logger.info(f"Out of scope request completed in {total_time:.2f}s")
                return {
                    "response": MessageSettings.ERROR_EXPERTISE_OUT,
                    "sources": []
                }
            
            if status == "SELAMLAMA":
                total_time = time.time() - start_time
                if AppSettings.PERFORMANCE_LOGGING:
                    logger.info(f"Greeting request completed in {total_time:.2f}s")
                return {
                    "response": "Merhaba! Ben bir üniversite tercih asistanıyım. Size YKS tercihleri, bölüm seçimi, kariyer planlaması konularında yardımcı olabilirim. Hangi konuda bilgi almak istiyorsunuz?",
                    "sources": []
                }
            
            # Adım 3: PARALEL İŞLEMLER - Enhanced question ile
            parallel_start = time.time()
            
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
                logger.error(f"Parallel processing error: {e}")
                context1 = "Vector search failed"
                context2 = "CSV analysis failed"
            
            parallel_time = time.time() - parallel_start
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"Parallel processing: {parallel_time:.2f}s")
            
            # Exception'ları handle et
            if isinstance(context1, Exception):
                logger.error(f"Vector context error: {context1}")
                context1 = "Vector search failed"
                
            if isinstance(context2, Exception):
                logger.error(f"CSV context error: {context2}")
                context2 = "CSV analysis failed"
            
            # Adım 4: Memory'den geçmiş al
            memory_start = time.time()
            conversation_history = self.memory.get_history(session_id)
            memory_time = time.time() - memory_start
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"Memory fetch: {memory_time:.3f}s")
            
            # Adım 5: Final yanıt oluşturma - Enhanced question ile
            final_start = time.time()
            final_response = await self._generate_final_response_safe(
                question=enhanced_question,  # Enhanced question kullan
                context1=context1,
                context2=context2,
                history=conversation_history
            )
            final_time = time.time() - final_start
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"Final response generation: {final_time:.2f}s")

            # Memory'ye kaydet - orijinal mesajı kaydet
            memory_save_start = time.time()
            self.memory.add_message(session_id, "user", message)  # Orijinal mesaj
            self.memory.add_message(session_id, "assistant", final_response)
            memory_save_time = time.time() - memory_save_start
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"Memory save: {memory_save_time:.3f}s")

            # PERFORMANS RAPORU
            total_time = time.time() - start_time
            
            if AppSettings.PERFORMANCE_LOGGING:
                if AppSettings.DETAILED_TIMING:
                    logger.info(f"Request completed in {total_time:.2f}s (Smart: {smart_time:.2f}s, Parallel: {parallel_time:.2f}s, Final: {final_time:.2f}s)")
                else:
                    logger.info(f"Request completed in {total_time:.2f}s")
            
            return {
                "response": final_response,
                "sources": self._extract_sources(context1, context2),
                "metadata": {
                    "processing_time": round(total_time, 2),
                    **({
                        "smart_evaluator_time": round(smart_time, 2),
                        "parallel_time": round(parallel_time, 2),
                        "enhanced_question": enhanced_question,
                        "original_question": message,
                        "status": status
                    } if AppSettings.DEBUG_MODE else {})
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Message processing error ({total_time:.2f}s): {e}")
            return {
                "response": MessageSettings.ERROR_GENERAL,
                "sources": [],
                "metadata": {"error": str(e), "processing_time": round(total_time, 2)}
            }

    async def _get_vector_context_native(self, question: str) -> str:
        """Native AstraDB ile vector arama - Temizlenmiş"""
        try:
            vector_start = time.time()
            
            if not self.astra_collection:
                logger.warning("Astra collection unavailable")
                return "Vector search unavailable"
            
            if AppSettings.LOG_VECTOR_DETAILS:
                logger.debug(f"Vector search starting: {question[:50]}...")
            
            # Embedding oluştur
            try:
                query_embedding = self.get_embedding(question)
                if AppSettings.LOG_VECTOR_DETAILS:
                    logger.debug(f"Query embedding created: {len(query_embedding)} dimensions")
            except Exception as e:
                logger.error(f"Embedding creation error: {e}")
                return "Embedding creation failed"
            
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
                
                if AppSettings.LOG_VECTOR_DETAILS:
                    logger.debug(f"Found {len(docs)} documents")
                
                if not docs:
                    logger.warning("No documents found")
                    return "No relevant documents found"
                
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
                        source = "Unknown source"
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
                            if not source_name or source_name == "Unknown source":
                                source_name = "Tercih Rehberi"
                        else:
                            source_name = "Rehber Dokümanı"
                        
                        context_parts.append(f"**Kaynak**: {source_name}\n**İçerik**: {content}")
                        total_chars += len(content)
                        
                        if AppSettings.LOG_VECTOR_DETAILS:
                            logger.debug(f"Document {i+1} processed: {source_name} - {len(content)} chars")
                        
                        if total_chars > 2000:
                            break
                            
                    except Exception as doc_error:
                        logger.error(f"Document {i+1} processing error: {doc_error}")
                        continue
                
                if not context_parts:
                    logger.error("No documents could be processed!")
                    return "Documents could not be processed"
                
                final_context = "\n\n".join(context_parts)
                vector_time = time.time() - vector_start
                
                if AppSettings.DETAILED_TIMING:
                    logger.debug(f"Vector search completed ({vector_time:.2f}s): {len(context_parts)} docs, {len(final_context)} chars")
                
                return final_context
                    
            except Exception as search_error:
                logger.error(f"Vector search error: {search_error}")
                return "Vector search failed"
            
        except Exception as e:
            vector_time = time.time() - vector_start
            logger.error(f"Vector context general error ({vector_time:.2f}s): {e}")
            return "Vector search general error"

    async def _get_csv_context_safe(self, question: str) -> str:
        """CSV analiz - Temizlenmiş"""
        try:
            csv_start = time.time()
            
            if self.csv_data is None:
                if AppSettings.LOG_CSV_DETAILS:
                    logger.debug("CSV data unavailable")
                return "CSV data unavailable"

            question_lower = question.lower()
            
            if AppSettings.LOG_CSV_DETAILS:
                logger.debug(f"CSV analysis starting: {question_lower[:50]}...")
            
            # CSV anahtar kelimesi kontrolü
            csv_keywords = [
                "istihdam", "maaş", "gelir", "sektör", "firma", "çalışma", "iş", 
                "girişim", "başlama", "oran", "yüzde", "istatistik", "veri",
                "bilgisayar", "mühendislik", "tıp", "hukuk", "ekonomi", "matematik",
                "fizik", "kimya", "makine", "elektrik", "endüstri"
            ]
            
            csv_required = any(keyword in question_lower for keyword in csv_keywords)
            
            if AppSettings.LOG_CSV_DETAILS:
                logger.debug(f"CSV keywords check: {csv_required}")
            
            if not csv_required:
                return "CSV analysis not required"

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
                if AppSettings.LOG_CSV_DETAILS:
                    logger.debug(f"Specific department analysis: {bolum_adi}")
                
                filtered = self.csv_data[self.csv_data['bolum_adi'] == bolum_adi]
                
                if filtered.empty:
                    return f"No data found for {bolum_adi}"
                
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
                if AppSettings.LOG_CSV_DETAILS:
                    logger.debug("General CSV analysis")
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
                        analysis = f"CSV analysis completed. Basic data for {bolum_adi or 'relevant departments'}: {csv_snippet[:200]}..."
                        
                except Exception as agent_error:
                    logger.error(f"CSV Agent error: {agent_error}")
                    analysis = f"CSV data found: {csv_snippet[:300]}..."
            else:
                analysis = f"CSV data: {csv_snippet[:300]}..."

            csv_time = time.time() - csv_start
            
            if AppSettings.DETAILED_TIMING:
                logger.debug(f"CSV analysis completed ({csv_time:.2f}s)")
            
            return analysis

        except Exception as e:
            csv_time = time.time() - csv_start
            logger.error(f"CSV analysis error ({csv_time:.2f}s): {e}")
            return "CSV analysis error"
            
    async def _generate_final_response_safe(self, question: str, context1: str, context2: str, history: str = "") -> str:
        """Final yanıt oluşturma - Temizlenmiş"""
        try:
            if not self.llm_final:
                logger.error("Final LLM unavailable!")
                return "Response generation service temporarily unavailable."
            
            result = await self.llm_final.ainvoke(
                self.final_prompt.format(
                    question=question,
                    context1=context1,
                    context2=context2,
                    history=history
                )
            )
            
            final_response = result.content.strip()
            
            if AppSettings.LOG_LLM_RESPONSES:
                logger.debug(f"Final response generated: {len(final_response)} chars")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Final response error: {e}")
            return "Error occurred while generating response."

    def _extract_sources(self, context1: str, context2: str) -> List[str]:
        """Kaynak çıkarma"""
        sources = []
        
        # Vector context kontrolü
        if context1 and len(context1.strip()) > 50:
            error_keywords = ["bulunamadı", "başarısız", "mevcut değil", "hata", "failed", "unavailable", "error"]
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
            csv_error_keywords = ["mevcut değil", "hata", "başarısız", "gerekli değil", "unavailable", "failed", "error", "not required"]
            has_csv_error = any(keyword in context2.lower() for keyword in csv_error_keywords)
            
            if not has_csv_error:
                csv_success_indicators = ["analiz", "oran", "veri", "bölüm", "istihdam", "maaş", "%", "analysis", "data"]
                has_csv_content = any(indicator in context2.lower() for indicator in csv_success_indicators)
                
                if has_csv_content:
                    sources.append(MessageSettings.SOURCES["UNIVERI_DB"])
        
        if not sources:
            sources.append(MessageSettings.SOURCES["GENERAL"])
        
        return list(dict.fromkeys(sources))

    async def test_all_connections(self) -> Dict[str, str]:
        """Bağlantı testleri"""
        if AppSettings.DEBUG_MODE:
            logger.debug("Testing all connections...")
        results = {}
        
        # OpenAI Client test
        try:
            if self.openai_client:
                test_embedding = self.get_embedding("test")
                results["OpenAI Client"] = f"✅ Connected ({len(test_embedding)} dimensions)"
            else:
                results["OpenAI Client"] = "❌ Client not initialized"
        except Exception as e:
            results["OpenAI Client"] = f"❌ Error: {str(e)[:50]}"
        
        # LLM testleri
        llm_tests = [
            ("Smart Evaluator-Corrector", self.llm_smart_evaluator_corrector),
            ("CSV Agent", self.llm_csv_agent),
            ("Final Response", self.llm_final)
        ]
        
        for name, llm in llm_tests:
            try:
                if llm:
                    await llm.ainvoke("Test")
                    results[name] = "✅ Connected"
                else:
                    results[name] = "❌ Model not loaded"
            except Exception as e:
                results[name] = f"❌ Error: {str(e)[:50]}"
        
        # AstraDB test
        try:
            if self.astra_collection:
                test_results = list(self.astra_collection.find({}, limit=1))
                results["AstraDB Native"] = f"✅ Connected ({len(test_results)} documents)"
            else:
                results["AstraDB Native"] = "❌ Collection not initialized"
        except Exception as e:
            results["AstraDB Native"] = f"❌ Error: {str(e)[:50]}"
        
        # CSV test
        try:
            if self.csv_data is not None:
                results["CSV"] = f"✅ Loaded ({len(self.csv_data)} rows)"
            else:
                results["CSV"] = "❌ Not loaded"
        except Exception as e:
            results["CSV"] = f"❌ Error: {str(e)[:50]}"
        
        # Memory test
        try:
            self.memory.add_message("test_connection", "user", "test")
            history = self.memory.get_history("test_connection")
            if history:
                results["Memory"] = "✅ Redis connected"
            else:
                results["Memory"] = "⚠️ Memory working but empty"
        except Exception as e:
            results["Memory"] = f"❌ Error: {str(e)[:50]}"
        
        return results
