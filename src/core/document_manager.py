"""
Document Manager - Handles document processing and storage
"""

import os
import sqlite3
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from config.settings import Settings
from utils.logger import setup_logger
from .text_processor import TextProcessor


class DocumentManager:
    """Manages document processing, storage, and retrieval"""
    
    def __init__(self):
        self.logger = setup_logger("DocumentManager")
        self.settings = Settings()
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.db_path = self.data_dir / "research_agent.db"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Initialize text processor
        self.text_processor = TextProcessor()
        
        # Initialize database
        self._init_database()
        
        # Document metadata
        self.metadata = {}
        self._load_metadata()
    
    def _init_database(self):
        """Initialize the document storage database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if documents table exists and get its schema
            cursor.execute("PRAGMA table_info(documents)")
            columns = cursor.fetchall()
            existing_columns = {col[1] for col in columns}
            
            if not existing_columns:
                # Create new table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT,
                        file_size INTEGER,
                        file_hash TEXT,
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN DEFAULT FALSE,
                        content_preview TEXT,
                        full_text_path TEXT
                    )
                """)
                self.logger.info("Created new documents table")
            else:
                # Add missing columns to existing table
                required_columns = {
                    'file_hash': 'TEXT',
                    'full_text_path': 'TEXT'
                }
                
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        cursor.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
                        self.logger.info(f"Added missing column: {col_name}")
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_metadata(self) -> None:
        """Load document metadata from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if file_hash column exists before querying
            cursor.execute("PRAGMA table_info(documents)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'file_hash' in columns:
                cursor.execute("SELECT filename, file_hash, processed FROM documents")
            else:
                cursor.execute("SELECT filename, '', processed FROM documents")
            
            rows = cursor.fetchall()
            
            for filename, file_hash, processed in rows:
                self.metadata[filename] = {
                    'hash': file_hash or '',
                    'processed': bool(processed)
                }
            
            conn.close()
            self.logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save document metadata to database"""
        try:
            self.logger.debug("Metadata saved to database")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for duplicate detection"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate if file can be processed"""
        try:
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            # Check supported file types
            supported_extensions = {'.pdf', '.txt', '.docx'}
            if file_path.suffix.lower() not in supported_extensions:
                return False, f"Unsupported file type: {file_path.suffix}"
            
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def is_duplicate(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if file is a duplicate based on hash"""
        try:
            file_hash = self._calculate_file_hash(file_path)
            if not file_hash:
                return False, None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if file_hash column exists
            cursor.execute("PRAGMA table_info(documents)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'file_hash' not in columns:
                return False, None
            
            cursor.execute("SELECT filename FROM documents WHERE file_hash = ?", (file_hash,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return True, result[0]
            return False, None
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {e}")
            return False, None
    
    def process_document(self, uploaded_file) -> bool:
        """Process uploaded file from Streamlit"""
        try:
            self.logger.info(f"Processing uploaded document: {uploaded_file.name}")
            print(f"🔄 PROCESSING FILE: {uploaded_file.name}")
            
            # Save uploaded file
            temp_path = self.uploads_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            print(f"✅ File saved to: {temp_path}")
            
            # Extract text content
            text_content = self._extract_text(temp_path)
            
            if not text_content or len(text_content.strip()) < 10:
                print(f"⚠️ Little text extracted from {uploaded_file.name}, using filename as content")
                text_content = f"Document: {uploaded_file.name}\nFile uploaded successfully but text extraction was minimal."
            
            print(f"📝 Extracted {len(text_content)} characters")
            
            # Save text content
            text_file_path = self.uploads_dir / f"{uploaded_file.name}_content.txt"
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"✅ Text content saved to: {text_file_path}")
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(temp_path)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO documents 
                (filename, file_type, file_size, file_hash, processed, content_preview, full_text_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                uploaded_file.name,
                temp_path.suffix.lower(),
                uploaded_file.size,
                file_hash,
                True,
                text_content[:500] + "..." if len(text_content) > 500 else text_content,
                str(text_file_path)
            ))
            
            conn.commit()
            conn.close()
            
            # Update metadata
            self.metadata[uploaded_file.name] = {
                'hash': file_hash,
                'processed': True
            }
            
            print(f"🎉 SUCCESS: {uploaded_file.name} processed completely!")
            self.logger.info(f"Successfully processed: {uploaded_file.name}")
            return True
                
        except Exception as e:
            error_msg = f"Error processing uploaded file: {e}"
            print(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return False
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from different file types"""
        file_type = file_path.suffix.lower()
        
        try:
            if file_type == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_type == '.txt':
                return self._extract_txt_text(file_path)
            elif file_type == '.docx':
                return self._extract_docx_text(file_path)
            else:
                return ""
        except Exception as e:
            self.logger.error(f"Text extraction failed for {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF"""
        print(f"📖 Extracting PDF text from: {file_path.name}")
        
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"📄 PDF has {len(pdf_reader.pages)} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"✅ Page {i+1}: {len(page_text)} characters")
                    else:
                        print(f"⚠️ Page {i+1}: No text found")
            
            print(f"✅ Total PDF text extracted: {len(text)} characters")
            return text.strip()
            
        except ImportError:
            print("❌ PyPDF2 not available, trying pdfplumber...")
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            print(f"✅ Page {i+1}: {len(page_text)} characters")
                return text.strip()
            except ImportError:
                print("❌ Neither PyPDF2 nor pdfplumber available")
                return ""
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"TXT extraction error: {e}")
            return ""
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except ImportError:
            self.logger.error("python-docx not available")
            return ""
        except Exception as e:
            self.logger.error(f"DOCX extraction error: {e}")
            return ""
    
    def search_documents(self, query: str, limit: int = 5) -> List[str]:
        """Search through processed documents"""
        try:
            print(f"🔍 SEARCHING for: '{query}'")
            
            results = []
            query_words = query.lower().split()
            
            # Get all content files
            content_files = list(self.uploads_dir.glob("*_content.txt"))
            print(f"📁 Found {len(content_files)} content files: {[f.name for f in content_files]}")
            
            if not content_files:
                return ["No processed documents found. Please upload and process some documents first."]
            
            for content_file in content_files:
                try:
                    with open(content_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content_lower = content.lower()
                    
                    # Check if query words are in content
                    found_words = [word for word in query_words if word in content_lower]
                    print(f"✅ Found words in {content_file.name}: {found_words}")
                    
                    if found_words:
                        # Find relevant sentences
                        sentences = content.split('.')
                        for sentence in sentences:
                            sentence_clean = sentence.strip()
                            if any(word in sentence.lower() for word in query_words) and len(sentence_clean) > 20:
                                # Clean up the filename for display
                                display_name = content_file.name.replace('_content.txt', '')
                                result = f"📄 **{display_name}**: {sentence_clean[:300]}{'...' if len(sentence_clean) > 300 else ''}"
                                results.append(result)
                                
                                if len(results) >= limit:
                                    break
                    
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    print(f"❌ Error searching {content_file}: {e}")
                    continue
            
            print(f"🎯 Found {len(results)} search results")
            return results[:limit] if results else [f"No relevant content found for '{query}' in your uploaded documents."]
            
        except Exception as e:
            error_msg = f"Document search error: {e}"
            print(f"❌ {error_msg}")
            return [error_msg]
    
    def sync_database_with_files(self):
        """Sync database records with actual files"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all database records
            cursor.execute("SELECT id, filename FROM documents")
            db_records = cursor.fetchall()
            
            removed_count = 0
            for doc_id, filename in db_records:
                # Check if files exist
                original_file = self.uploads_dir / filename
                content_file = self.uploads_dir / f"{filename}_content.txt"
                base_name = Path(filename).stem
                alt_content_file = self.uploads_dir / f"{base_name}_content.txt"
                
                files_exist = any([
                    original_file.exists(),
                    content_file.exists(),
                    alt_content_file.exists()
                ])
                
                if not files_exist:
                    # Remove from database
                    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
                    removed_count += 1
                    self.logger.info(f"Removed orphaned record: {filename}")
            
            conn.commit()
            conn.close()
            
            if removed_count > 0:
                self.logger.info(f"Database sync complete - removed {removed_count} orphaned records")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Database sync failed: {e}")
            return 0

    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get list of processed documents with auto-sync"""
        try:
            # First, sync database with actual files
            self.sync_database_with_files()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, file_size, upload_date, status
                FROM documents
                ORDER BY upload_date DESC
            ''')
            
            documents = []
            for row in cursor.fetchall():
                # Verify file still exists before adding to list
                filename = row[1]
                content_file = self.uploads_dir / f"{filename}_content.txt"
                base_name = Path(filename).stem  
                alt_content_file = self.uploads_dir / f"{base_name}_content.txt"
                
                if content_file.exists() or alt_content_file.exists():
                    documents.append({
                        'id': row[0],
                        'filename': row[1],
                        'file_size': row[2] or 0,
                        'upload_date': row[3] or 'Unknown',
                        'status': row[4] or 'ready'
                    })
            
            conn.close()
            return documents
            
        except Exception as e:
            self.logger.error(f"Error getting processed documents: {e}")
            return []
    
    def is_healthy(self) -> bool:
        """Check if DocumentManager is functioning properly"""
        try:
            return (
                self.uploads_dir.exists() and 
                Path(self.db_path).exists()
            )
        except:
            return False
