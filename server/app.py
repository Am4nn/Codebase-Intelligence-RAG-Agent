"""
FastAPI Application

REST API server for Codebase Intelligence.

Endpoints:
    POST /query - Query the codebase
    GET /health - Health check
    GET /status - System status
    POST /export - Export change log
"""
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.requests import Request
from core import CodebaseIntelligence

def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the server."""
    # Configure logging at module level to ensure it's set before uvicorn starts
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True  # Force reconfiguration even if logging is already configured
    )
    
    # Explicitly set log level for core modules to ensure they're visible
    logging.getLogger("core").setLevel(level)
    logging.getLogger("server").setLevel(level)

configure_logging()

logger = logging.getLogger(__name__)

# Global system instance
system: Optional[CodebaseIntelligence] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event handler for startup and shutdown."""
    global system
    
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 CODEBASE INTELLIGENCE SERVER STARTING")
    logger.info("=" * 70)
    
    try:
        # Initialize system
        system = CodebaseIntelligence()

        logger.info("⚙️  Initializing core system...")
        await system.initialize()

        logger.info("✅ Server ready!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        # Continue anyway - API will return 503
    
    yield
    
    # Shutdown
    logger.info("👋 Server shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Codebase Intelligence API",
    description="REST API for querying and analyzing codebases using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for querying the codebase."""
    question: str = Field(..., description="Natural language question about the codebase")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context tracking")

class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str = Field(..., description="Answer from the AI agent")
    question: str = Field(..., description="Original question")

class StatusResponse(BaseModel):
    """Response model for system status."""
    initialized: bool = Field(..., description="Whether the system is initialized")
    repo_path: str = Field(..., description="Path to the repository")
    persist_dir: str = Field(..., description="Path to vector store")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Server health status")
    system_ready: bool = Field(..., description="Whether the system is ready")

class ExportResponse(BaseModel):
    """Response model for export operation."""
    success: bool = Field(..., description="Whether export was successful")
    file_path: str = Field(..., description="Path to exported file")

class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str = Field(..., description="The conversation ID")
    messages: list = Field(..., description="List of messages in the conversation")
    message_count: int = Field(..., description="Total number of messages")

class ConversationListResponse(BaseModel):
    """Response model for listing conversations."""
    conversations: list = Field(..., description="List of conversation IDs")
    count: int = Field(..., description="Total number of conversations")

class ConversationStateResponse(BaseModel):
    """Response model for conversation state."""
    conversation_id: str = Field(..., description="The conversation ID")
    exists: bool = Field(..., description="Whether the conversation exists")
    state: Optional[dict] = Field(None, description="Full conversation state")

class ConversationSummaryResponse(BaseModel):
    """Response model for conversation summary."""
    conversation_id: str = Field(..., description="The conversation ID")
    exists: bool = Field(..., description="Whether the conversation exists")
    message_count: int = Field(..., description="Total number of messages")
    role_counts: Optional[dict] = Field(None, description="Message count by role")
    first_message: Optional[str] = Field(None, description="First message preview")
    last_message: Optional[str] = Field(None, description="Last message preview")

class ClearConversationResponse(BaseModel):
    """Response model for clearing a conversation."""
    success: bool = Field(..., description="Whether the operation was successful")
    conversation_id: str = Field(..., description="The conversation ID that was cleared")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": "Codebase Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "validEndpoints": [
            "GET /health",
            "GET /status",
            "POST /query",
            "POST /export",
            "GET /conversations",
            "GET /conversations/{conversation_id}/history",
            "GET /conversations/{conversation_id}/state",
            "GET /conversations/{conversation_id}/summary",
            "DELETE /conversations/{conversation_id}"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the server health status and whether the system is ready.
    """
    return HealthResponse(
        status="healthy",
        system_ready=system is not None and system.is_initialized()
    )

@app.get("/status", response_model=StatusResponse, tags=["General"])
async def get_status() -> StatusResponse:
    """
    Get system status.
    
    Returns information about the initialized system.
    """
    if system is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    return StatusResponse(
        initialized=system.is_initialized(),
        repo_path=system.repo_path,
        persist_dir=system.persist_dir
    )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_codebase(request: QueryRequest) -> QueryResponse:
    """
    Query the codebase with a natural language question.
    
    Args:
        request: Query request containing the question
        
    Returns:
        Answer from the AI agent
        
    Raises:
        HTTPException: If system not initialized or query fails
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please wait for initialization to complete."
        )
    
    try:
        logger.info(f"🔍 Query: {request.question[:100]}...")
        answer = await system.query(request.question, request.conversation_id or "default")
        logger.info(f"✅ Query completed")
        
        return QueryResponse(
            answer=answer,
            question=request.question
        )
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

@app.post("/export", response_model=ExportResponse, tags=["Export"])
async def export_change_log(background_tasks: BackgroundTasks, output_file: str = "change_log.json") -> ExportResponse:
    """
    Export the agent's change log to a file.
    
    Args:
        output_file: Path to output file (default: change_log.json)
        
    Returns:
        Success status and file path
        
    Raises:
        HTTPException: If system not initialized or export fails
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        system.export_change_log(output_file)
        logger.info(f"✅ Exported change log to {output_file}")
        
        return ExportResponse(
            success=True,
            file_path=output_file
        )
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )

@app.get("/conversations", response_model=ConversationListResponse, tags=["Conversations"])
async def list_conversations() -> ConversationListResponse:
    """
    List all conversation IDs.
    
    Returns:
        List of all conversation IDs stored in the system
        
    Raises:
        HTTPException: If system not initialized
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        conversations = await system.list_conversations()
        return ConversationListResponse(
            conversations=conversations,
            count=len(conversations)
        )
    except Exception as e:
        logger.error(f"❌ Failed to list conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list conversations: {str(e)}"
        )

@app.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse, tags=["Conversations"])
async def get_conversation_history(conversation_id: str) -> ConversationHistoryResponse:
    """
    Get the full conversation history for a specific conversation ID.
    
    Args:
        conversation_id: The conversation ID to retrieve history for
        
    Returns:
        Full conversation history with all messages
        
    Raises:
        HTTPException: If system not initialized or conversation not found
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        messages = await system.get_conversation_history(conversation_id)
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            message_count=len(messages)
        )
    except Exception as e:
        logger.error(f"❌ Failed to get conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )

@app.get("/conversations/{conversation_id}/state", response_model=ConversationStateResponse, tags=["Conversations"])
async def get_conversation_state(conversation_id: str) -> ConversationStateResponse:
    """
    Get the complete state/checkpoint for a conversation.
    
    Args:
        conversation_id: The conversation ID to retrieve state for
        
    Returns:
        Full conversation state including metadata
        
    Raises:
        HTTPException: If system not initialized
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        state = await system.get_conversation_state(conversation_id)
        return ConversationStateResponse(
            conversation_id=conversation_id,
            exists=state is not None,
            state=state
        )
    except Exception as e:
        logger.error(f"❌ Failed to get conversation state: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation state: {str(e)}"
        )

@app.get("/conversations/{conversation_id}/summary", response_model=ConversationSummaryResponse, tags=["Conversations"])
async def get_conversation_summary(conversation_id: str) -> ConversationSummaryResponse:
    """
    Get a summary of a conversation including message count and participants.
    
    Args:
        conversation_id: The conversation ID to summarize
        
    Returns:
        Conversation summary with statistics
        
    Raises:
        HTTPException: If system not initialized
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        summary = await system.get_conversation_summary(conversation_id)
        return ConversationSummaryResponse(**summary)
    except Exception as e:
        logger.error(f"❌ Failed to get conversation summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation summary: {str(e)}"
        )

@app.delete("/conversations/{conversation_id}", response_model=ClearConversationResponse, tags=["Conversations"])
async def clear_conversation(conversation_id: str) -> ClearConversationResponse:
    """
    Clear/delete a specific conversation history.
    
    Args:
        conversation_id: The conversation ID to clear
        
    Returns:
        Success status
        
    Raises:
        HTTPException: If system not initialized or clear operation fails
    """
    if system is None or not system.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    
    try:
        success = await system.clear_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear conversation"
            )
        
        logger.info(f"✅ Cleared conversation: {conversation_id}")
        return ClearConversationResponse(
            success=success,
            conversation_id=conversation_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to clear conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint does not exist",
            "docs": "/docs"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 500 errors."""
    logger.exception("Internal server error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred"
        }
    )
