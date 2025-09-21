"""
Text Processor - Handle document parsing, text extraction, and chunking
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Document processing imports
import pdfplumber
import PyPDF2
from docx import Document as DocxDocument
import markdown

from config.settings import Settings
from config.model_config import ModelConfig
from utils.logger import logger


@dataclass
class TextChunk:
    """Data class for text chunks"""
    content: str
    chunk_id: str
    doc_id: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        """Calculate statistics after initialization"""
        self.word_count = len(self.content.split())
        self.char_count = len(self.content)


class TextProcessor:
    """Handles text extraction and chunking from various document formats"""
    
    def __init__(self):
        """Initialize text processor"""
        self.settings = Settings()
        self.model_config = ModelConfig()
        
    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from PDF using pdfplumber (primary) and PyPDF2 (fallback)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        text = ""
        metadata = {"pages": 0, "extraction_method": ""}
        
        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                metadata["extraction_method"] = "pdfplumber"
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n"
                        text += page_text + "\n"
                        
            if text.strip():
                return text, metadata
                
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                metadata["extraction_method"] = "PyPDF2"
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n"
                        text += page_text + "\n"
                        
        except Exception as e:
            logger.error(f"PyPDF2 also failed for {file_path}: {e}")
            return "", {"error": str(e)}
        
        return text, metadata
    
    def extract_text_from_docx(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = DocxDocument(file_path)
            text = ""
            paragraph_count = 0
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
                    paragraph_count += 1
            
            metadata = {
                "paragraphs": paragraph_count,
                "extraction_method": "python-docx"
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return "", {"error": str(e)}
    
    def extract_text_from_txt(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from plain text file
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                        
                    metadata = {
                        "encoding": encoding,
                        "extraction_method": "direct_read"
                    }
                    
                    return text, metadata
                    
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail
            logger.error(f"Could not decode text file {file_path}")
            return "", {"error": "Could not decode file with any encoding"}
            
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return "", {"error": str(e)}
    
    def extract_text_from_markdown(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from Markdown file
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # Convert markdown to plain text (remove markdown syntax)
            html = markdown.markdown(md_content)
            
            # Simple HTML tag removal (basic approach)
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up extra whitespace
            
            metadata = {
                "original_markdown_length": len(md_content),
                "extraction_method": "markdown_parser"
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from Markdown {file_path}: {e}")
            return "", {"error": str(e)}
    
    def extract_text(self, file_path: Path) -> Tuple[str, Dict]:
        """
        Extract text from any supported document format
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        elif file_ext == '.md':
            return self.extract_text_from_markdown(file_path)
        else:
            error_msg = f"Unsupported file format: {file_ext}"
            logger.error(error_msg)
            return "", {"error": error_msg}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        # Fix common OCR/extraction errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after punctuation
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Normalize quotes - FIXED REGEX PATTERNS
        text = re.sub(r'["“”]+', '"', text)  # Normalize double quotes (including curly)
        text = re.sub(r"[’‘']+", "'", text)  # Normalize single quotes (including curly)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks(self, text: str, doc_id: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks for embedding
        
        Args:
            text: Cleaned text to chunk
            doc_id: Document ID
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        chunk_size = self.settings.CHUNK_SIZE
        overlap_size = self.settings.CHUNK_OVERLAP
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        doc_id=doc_id,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if len(current_chunk) > overlap_size:
                    overlap_text = current_chunk[-overlap_size:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def process_document(self, file_path: Path, doc_id: str) -> Tuple[List[TextChunk], Dict]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to document
            doc_id: Document ID
            
        Returns:
            Tuple of (chunks, processing_metadata)
        """
        logger.info(f"Processing document: {doc_id}")
        
        # Extract text
        raw_text, extraction_metadata = self.extract_text(file_path)
        
        if not raw_text or "error" in extraction_metadata:
            logger.error(f"Failed to extract text from {doc_id}")
            return [], extraction_metadata
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Create chunks
        chunks = self.create_chunks(cleaned_text, doc_id)
        
        # Prepare processing metadata
        processing_metadata = {
            **extraction_metadata,
            "raw_text_length": len(raw_text),
            "cleaned_text_length": len(cleaned_text),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk.char_count for chunk in chunks) / len(chunks) if chunks else 0,
            "processing_status": "success"
        }
        
        logger.info(f"Successfully processed {doc_id}: {len(chunks)} chunks created")
        return chunks, processing_metadata
    
    def save_chunks(self, chunks: List[TextChunk], doc_id: str) -> bool:
        """
        Save chunks to disk for later retrieval
        
        Args:
            chunks: List of text chunks
            doc_id: Document ID
            
        Returns:
            Success status
        """
        try:
            chunks_file = self.settings.DATA_DIR / f"{doc_id}_chunks.json"
            
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                    "char_count": chunk.char_count
                })
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks for {doc_id}: {e}")
            return False
    
    def load_chunks(self, doc_id: str) -> List[TextChunk]:
        """
        Load chunks from disk
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of TextChunk objects
        """
        try:
            chunks_file = self.settings.DATA_DIR / f"{doc_id}_chunks.json"
            
            if not chunks_file.exists():
                return []
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = []
            for chunk_data in chunks_data:
                chunk = TextChunk(
                    content=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    doc_id=chunk_data["doc_id"],
                    page_number=chunk_data.get("page_number"),
                    chunk_index=chunk_data["chunk_index"]
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks for {doc_id}: {e}")
            return []