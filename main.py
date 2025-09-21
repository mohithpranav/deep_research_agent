"""
Deep Researcher Agent - Main Application Entry Point
Uses modular architecture from src/ folder
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Page Configuration
st.set_page_config(
    page_title="Deep Researcher Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    try:
        # Import essential modules
        from utils.logger import setup_logger
        from core.document_manager import DocumentManager
        
        # Setup logging
        logger = setup_logger("DeepResearcherAgent")
        logger.info("Starting Deep Researcher Agent")
        
        # App header
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h1>🔬 Deep Researcher Agent</h1>
            <p>AI-Powered Document Analysis & Research Assistant</p>
            <p><small>Upload any document • Ask any question • Get deep insights</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize document manager
        doc_manager = DocumentManager()
        
        # Initialize session state for processed files
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []
        
        # Sidebar for file upload
        with st.sidebar:
            st.subheader("📄 Document Upload")
            
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'txt', 'docx'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or DOCX files for analysis"
            )
            
            # Process uploaded files immediately when uploaded
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check if file is already processed
                    if uploaded_file.name not in st.session_state.processed_files:
                        st.write(f"🔄 Processing: {uploaded_file.name}")
                        
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            success = doc_manager.process_document(uploaded_file)
                            
                            if success:
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                st.session_state.processed_files.append(uploaded_file.name)
                                
                                # Show file details
                                st.info(f"📊 Size: {uploaded_file.size:,} bytes")
                            else:
                                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Show processed documents
            st.subheader("📚 Processed Documents")
            processed_docs = doc_manager.get_processed_documents()
            
            if processed_docs:
                st.write(f"**Total Documents: {len(processed_docs)}**")
                for doc in processed_docs:
                    st.text(f"📄 {doc['filename']}")
                    
                # Add debug button
                if st.button("🔍 Debug File Storage"):
                    uploads_dir = project_root / "data" / "uploads"
                    if uploads_dir.exists():
                        all_files = list(uploads_dir.iterdir())
                        content_files = list(uploads_dir.glob("*_content.txt"))
                        
                        st.write(f"📁 **Upload directory:** {uploads_dir}")
                        st.write(f"📄 **All files:** {len(all_files)}")
                        st.write(f"📝 **Content files:** {len(content_files)}")
                        
                        for file in all_files:
                            st.text(f"  • {file.name}")
                    else:
                        st.error("Upload directory doesn't exist!")
            else:
                st.info("No documents processed yet")
            
            # System status
            st.subheader("🔧 System Status")
            
            # Test LLM connection
            try:
                from reasoning.llm_client import LLMClient
                llm_client = LLMClient()
                llm_healthy = llm_client.is_healthy()
                st.markdown(f"**🧠 AI Model (llama2:7b)**: {'✅ Connected' if llm_healthy else '❌ Offline'}")
            except Exception as e:
                st.markdown(f"**🧠 AI Model**: ❌ Error - {str(e)[:50]}...")
            
            doc_healthy = doc_manager.is_healthy()
            st.markdown(f"**📄 Document Manager**: {'✅ Healthy' if doc_healthy else '❌ Issues'}")
            
            uploads_dir = project_root / "data" / "uploads"
            db_path = project_root / "data" / "research_agent.db"
            st.text(f"📁 Uploads: {'✅' if uploads_dir.exists() else '❌'}")
            st.text(f"💾 Database: {'✅' if db_path.exists() else '❌'}")
        
        # Main chat interface
        st.subheader("💬 AI-Powered Document Q&A")
        
        # Show current file status
        uploads_dir = project_root / "data" / "uploads"
        content_files = list(uploads_dir.glob("*_content.txt")) if uploads_dir.exists() else []
        st.info(f"🧠 **AI Ready:** llama2:7b connected | 📚 **Documents:** {len(content_files)} processed")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask intelligent questions about your documents..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔬 AI is analyzing your question and researching documents..."):
                    
                    # Import reasoning modules with proper error handling
                    try:
                        from reasoning.response_synthesizer import ResponseSynthesizer
                        from reasoning.query_analyzer import QueryAnalyzer
                        from reasoning.llm_client import LLMClient
                        
                        # Initialize reasoning components
                        synthesizer = ResponseSynthesizer()
                        query_analyzer = QueryAnalyzer()
                        llm_client = LLMClient()
                        
                        # Check if AI is available
                        if llm_client.is_healthy():
                            st.success("🧠 AI Model Connected - Generating intelligent response...")
                            
                            # Analyze the query
                            query_info = query_analyzer.analyze(prompt)
                            
                            # Search documents for context
                            search_results = doc_manager.search_documents(prompt, limit=5)
                            
                            # Use AI to synthesize comprehensive response
                            response = synthesizer.synthesize(
                                query=prompt,
                                context=search_results,
                                query_info=query_info
                            )
                        else:
                            st.warning("🔧 AI Model offline - Using basic search...")
                            raise Exception("AI model not available")
                            
                    except Exception as e:
                        st.info("🔍 Using basic document search (AI model unavailable)")
                        
                        # Fallback to basic response
                        search_results = doc_manager.search_documents(prompt, limit=5)
                        
                        if search_results and not any(result.startswith("No") for result in search_results):
                            response = "**Based on your documents, here's what I found:**\n\n"
                            for i, result in enumerate(search_results, 1):
                                response += f"**Finding {i}:**\n{result}\n\n"
                            response += "\n💡 *For AI-powered comprehensive analysis, ensure Ollama is running: `ollama serve`*"
                        else:
                            if content_files:
                                response = f"I searched through {len(content_files)} processed documents but couldn't find specific information about '{prompt}'. The documents might not contain this information, or try rephrasing your question."
                            else:
                                response = "No processed documents found. Please upload documents first using the sidebar."
                    
                    st.write(response)
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
        
        # Footer with controls
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Show System Info"):
                st.info(f"**Documents:** {len(processed_docs)} | **Content Files:** {len(content_files)}")
                st.code(f"Model: llama2:7b\nOllama: http://localhost:11434")
        
        with col3:
            # Show live status
            try:
                from reasoning.llm_client import LLMClient
                llm_status = "🟢" if LLMClient().is_healthy() else "🔴"
            except:
                llm_status = "🔴"
            st.metric("🧠 AI Status", llm_status)
        
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.exception(e)
        st.info("Please check your dependencies and module imports")

if __name__ == "__main__":
    main()

