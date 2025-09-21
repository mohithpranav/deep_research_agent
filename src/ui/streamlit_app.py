"""
Deep Researcher Agent - Streamlit Web Application
"""
import streamlit as st
import sqlite3
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import our core components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from config.model_config import ModelConfig
from core.document_manager import DocumentManager
from core.text_processor import TextProcessor
from core.embedding_engine import EmbeddingEngine
from storage.vector_store import VectorStore
from reasoning.query_analyzer import QueryAnalyzer
from reasoning.multi_step_reasoner import MultiStepReasoner
from reasoning.response_synthesizer import ResponseSynthesizer
from utils.logger import logger


# Page configuration
st.set_page_config(
    page_title="Deep Researcher Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        padding: 1rem;
    }
    
    /* Chat bubble styling */
    .user-message {
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0;
        margin-left: 20%;
        border-left: 3px solid #9ca3af;
    }
    
    .ai-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 3px solid #2196f3;
    }
    
    /* Document card styling */
    .doc-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #28a745;
    }
    
    /* Settings panel styling */
    .settings-panel {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Source box styling */
    .source-box {
        background-color: #fff3cd;
        border-radius: 5px;
        padding: 8px;
        margin: 5px 0;
        border-left: 3px solid #ffc107;
        font-size: 0.9em;
    }
    
    /* Status indicator styling */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


class ChatDatabase:
    """Manages SQLite database for chat history"""
    
    def __init__(self, db_path: str = "data/chat_history.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                sources_count INTEGER DEFAULT 0,
                processing_time REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_conversation(self, query: str, response: str, confidence: float = 0.0, 
                        sources_count: int = 0, processing_time: float = 0.0):
        """Add a conversation to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (query, response, confidence, sources_count, processing_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response, confidence, sources_count, processing_time))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """Get recent chat history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, query, response, confidence, sources_count, processing_time
            FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "query": row[1],
                "response": row[2],
                "confidence": row[3],
                "sources_count": row[4],
                "processing_time": row[5]
            }
            for row in results
        ]
    
    def clear_history(self):
        """Clear all chat history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_history')
        conn.commit()
        conn.close()


def initialize_components():
    """Initialize all core components"""
    if 'components_initialized' not in st.session_state:
        with st.spinner("🔧 Initializing Deep Researcher Agent..."):
            try:
                # Initialize core components
                st.session_state.settings = Settings()
                st.session_state.doc_manager = DocumentManager()
                st.session_state.text_processor = TextProcessor()
                st.session_state.embedding_engine = EmbeddingEngine()
                st.session_state.vector_store = VectorStore()
                st.session_state.query_analyzer = QueryAnalyzer()
                st.session_state.reasoner = MultiStepReasoner(
                    st.session_state.vector_store,
                    st.session_state.embedding_engine
                )
                st.session_state.synthesizer = ResponseSynthesizer()
                st.session_state.chat_db = ChatDatabase()
                
                st.session_state.components_initialized = True
                st.success("✅ All components initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Initialization error: {str(e)}")
                st.session_state.components_initialized = False


def process_query(query: str, top_k: int = 5) -> Dict:
    """Process user query through the complete pipeline"""
    start_time = time.time()
    
    try:
        # Step 1: Analyze query
        with st.spinner("🧠 Analyzing your question..."):
            query_intent = st.session_state.query_analyzer.analyze_query(query)
        
        # Step 2: Multi-step reasoning
        with st.spinner("🔍 Performing multi-step reasoning..."):
            reasoning_result = st.session_state.reasoner.execute_reasoning_chain(query_intent)
        
        # Step 3: Synthesize response
        with st.spinner("📝 Synthesizing final response..."):
            formatted_response = st.session_state.synthesizer.synthesize_response(
                reasoning_result, 
                format_style="conversational"
            )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "response": formatted_response.answer,
            "confidence": formatted_response.confidence_score,
            "sources_count": formatted_response.sources_count,
            "processing_time": processing_time,
            "citations": formatted_response.citations,
            "reasoning_steps": reasoning_result.reasoning_steps
        }
        
    except Exception as e:
        return {
            "success": False,
            "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
            "confidence": 0.0,
            "sources_count": 0,
            "processing_time": time.time() - start_time,
            "citations": [],
            "reasoning_steps": []
        }


def handle_file_upload(uploaded_file):
    """Handle document upload and processing"""
    if uploaded_file is not None:
        try:
            # Save uploaded file
            file_path = Path("data/uploads") / uploaded_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add document to system
            success, result = st.session_state.doc_manager.add_document(file_path)
            
            if success:
                doc_id = result
                st.success(f"✅ Document uploaded: {uploaded_file.name}")
                
                # Process document
                with st.spinner("📄 Processing document..."):
                    doc_path = st.session_state.doc_manager.get_document_path(doc_id)
                    chunks, metadata = st.session_state.text_processor.process_document(doc_path, doc_id)
                    
                    if chunks:
                        # Generate embeddings and store
                        embeddings, chunk_ids = st.session_state.embedding_engine.encode_chunks(chunks)
                        st.session_state.vector_store.add_chunks(chunks, embeddings)
                        
                        # Mark as processed
                        st.session_state.doc_manager.mark_processed(doc_id)
                        st.session_state.doc_manager.mark_embedding_generated(doc_id)
                        
                        st.success(f"✅ Document processed: {len(chunks)} chunks created")
                    else:
                        st.error("❌ Failed to process document")
                        
            else:
                st.error(f"❌ Upload failed: {result}")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


def render_chat_bubble(message: str, is_user: bool = False, metadata: Dict = None):
    """Render a chat message bubble"""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_indicator = ""
        sources_indicator = ""
        
        if metadata:
            confidence = metadata.get("confidence", 0)
            sources_count = metadata.get("sources_count", 0)
            confidence_indicator = f" (Confidence: {confidence:.1%})"
            if sources_count > 0:
                sources_indicator = f" • {sources_count} sources"
        
        st.markdown(f"""
        <div class="ai-message">
            <strong>🔬 Deep Researcher:</strong>{confidence_indicator}{sources_indicator}<br>
            {message}
        </div>
        """, unsafe_allow_html=True)


def render_document_card(doc_info: Dict):
    """Render a document information card"""
    doc_name = doc_info.get("custom_name", doc_info.get("original_name", "Unknown"))
    doc_size = doc_info.get("file_size", 0)
    upload_date = doc_info.get("upload_date", "Unknown")
    processed = doc_info.get("processed", False)
    
    # Format file size
    if doc_size > 1024 * 1024:
        size_str = f"{doc_size / (1024 * 1024):.1f} MB"
    elif doc_size > 1024:
        size_str = f"{doc_size / 1024:.1f} KB"
    else:
        size_str = f"{doc_size} bytes"
    
    status_color = "#28a745" if processed else "#ffc107"
    status_text = "✅ Processed" if processed else "⏳ Processing"
    
    st.markdown(f"""
    <div class="doc-card">
        <strong>{doc_name}</strong><br>
        <small>Size: {size_str} • Uploaded: {upload_date[:10]}</small><br>
        <span style="color: {status_color};">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)


def render_source_boxes(citations):
    """Render source citation boxes"""
    if not citations:
        return
    
    st.markdown("**📚 Sources:**")
    for citation in citations[:3]:  # Show top 3 sources
        st.markdown(f"""
        <div class="source-box">
            <strong>[{citation.id}] {citation.doc_id}</strong><br>
            <small>{citation.content_snippet}</small>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("# 🔬 Deep Researcher Agent")
    st.markdown("*Advanced multi-step reasoning for document analysis*")
    st.markdown("---")
    
    # Initialize components
    initialize_components()
    
    if not st.session_state.get('components_initialized', False):
        st.error("❌ System not initialized. Please refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Top-k retrieval slider
        top_k = st.slider(
            "📊 Retrieved Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve for analysis"
        )
        
        st.markdown("---")
        
        # File uploader
        st.markdown("## 📁 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md', 'docx'],
            help="Supported formats: PDF, TXT, Markdown, DOCX"
        )
        
        if uploaded_file:
            handle_file_upload(uploaded_file)
        
        st.markdown("---")
        
        # System status
        st.markdown("## 🔧 System Status")
        
        # Get system stats
        if hasattr(st.session_state, 'vector_store'):
            stats = st.session_state.vector_store.get_stats()
            st.metric("Documents", stats.get('active_chunks', 0))
            st.metric("Vector Database", f"{stats.get('total_vectors', 0)} vectors")
        
        # LLM status
        if hasattr(st.session_state, 'reasoner'):
            llm_info = st.session_state.reasoner.llm_manager.get_provider_info()
            llm_status = "🟢 Online" if llm_info['available'] else "🔴 Offline"
            st.markdown(f"**LLM Status:** {llm_status}")
            if llm_info['model']:
                st.markdown(f"**Model:** {llm_info['model']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    # Left column - Chat interface
    with col1:
        st.markdown("## 💬 Research Chat")
        
        # Query input
        query = st.text_area(
            "Ask me anything about your uploaded documents:",
            height=100,
            placeholder="e.g., 'Compare the main findings from the research papers' or 'What are the key themes across all documents?'"
        )
        
        col1_1, col1_2 = st.columns([1, 4])
        with col1_1:
            submit_button = st.button("🚀 Submit", type="primary")
        
        # Process query
        if submit_button and query.strip():
            # Add user message
            render_chat_bubble(query, is_user=True)
            
            # Process and display response
            result = process_query(query, top_k)
            
            if result["success"]:
                metadata = {
                    "confidence": result["confidence"],
                    "sources_count": result["sources_count"]
                }
                render_chat_bubble(result["response"], metadata=metadata)
                
                # Show sources if available
                if result.get("citations"):
                    render_source_boxes(result["citations"])
                
                # Save to database
                st.session_state.chat_db.add_conversation(
                    query=query,
                    response=result["response"],
                    confidence=result["confidence"],
                    sources_count=result["sources_count"],
                    processing_time=result["processing_time"]
                )
            else:
                render_chat_bubble(result["response"])
        
        # Display reasoning steps if available
        if submit_button and query.strip() and 'reasoning_steps' in result:
            with st.expander("🧠 View Reasoning Steps", expanded=False):
                for i, step in enumerate(result.get('reasoning_steps', []), 1):
                    st.markdown(f"**Step {i}:** {step.question}")
                    st.markdown(f"*Response:* {step.response[:200]}...")
                    st.markdown("---")
    
    # Right column - Chat history and documents
    with col2:
        st.markdown("## 📚 Document Library")
        
        # List uploaded documents
        docs = st.session_state.doc_manager.list_documents()
        
        if docs:
            for doc_id, doc_info in list(docs.items())[:5]:  # Show recent 5 docs
                render_document_card(doc_info)
        else:
            st.info("No documents uploaded yet. Use the sidebar to upload documents.")
        
        st.markdown("---")
        st.markdown("## 🕒 Chat History")
        
        # Display chat history
        chat_history = st.session_state.chat_db.get_chat_history(limit=5)
        
        if chat_history:
            for chat in chat_history:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 8px; margin: 5px 0; border-radius: 5px; font-size: 0.9em;">
                    <strong>Q:</strong> {chat['query'][:50]}...
                    <br><small>{chat['timestamp'][:16]} • {chat['confidence']:.1%} confidence</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_db.clear_history()
                st.experimental_rerun()
        else:
            st.info("No chat history yet. Start asking questions!")


if __name__ == "__main__":
    main()