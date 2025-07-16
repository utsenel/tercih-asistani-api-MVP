@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest): 
    import time
    start_time = time.time()
    
    try:
        # SESSION ID YÖNETİMİ - Sadeleştirilmiş
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        real_ip = request.headers.get("X-Real-IP") 
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")
        
        actual_ip = cf_connecting_ip or real_ip or forwarded_for or client_ip
        if forwarded_for and "," in forwarded_for:
            actual_ip = forwarded_for.split(",")[0].strip()
        
        original_session_id = chat_request.session_id
        
        if not chat_request.session_id or chat_request.session_id in ["ng", "default", ""]:
            user_agent = request.headers.get("user-agent", "unknown")[:50]
            hash_input = f"{actual_ip}_{user_agent}"
            ip_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            chat_request.session_id = f"stable_{ip_hash}"
        
        # SADECE ÖNEMLİ LOGLARI TUTALIM
        if AppSettings.LOG_SESSION_DETAILS:
            logger.debug(f"Session: {original_session_id} → {chat_request.session_id}")
            logger.debug(f"Client IP: {actual_ip}")
        
        logger.info(f"Chat request: session={chat_request.session_id}, message_len={len(chat_request.message)}")
        
        # Chat processor ile işle
        result = await processor.process_message(
            message=chat_request.message,
            session_id=chat_request.session_id
        )
        
        # Processing time hesapla
        processing_time = round(time.time() - start_time, 2)
        
        # SADECE GEREKLİ PERFORMANCE LOG
        if AppSettings.PERFORMANCE_LOGGING:
            logger.info(f"Request completed in {processing_time}s")
        
        return ChatResponse(
            response=result["response"],
            status="success",
            sources=result.get("sources", []),
            metadata={
                "processing_time": processing_time,
                "session_id": chat_request.session_id,
                "message_length": len(chat_request.message),
                **({
                    "client_ip": actual_ip,
                    "session_transition": f"{original_session_id} → {chat_request.session_id}"
                } if AppSettings.DEBUG_MODE else {})
            }
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"Chat endpoint error: {str(e)}")
        
        return ChatResponse(
            response="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            status="error",
            sources=[],
            metadata={
                "processing_time": processing_time,
                "error": str(e)[:100]
            }
        )
