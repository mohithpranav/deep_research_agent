"""
Settings and configuration for the Deep Researcher Agent
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class Settings:
    """Configuration settings for the application"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.uploads_dir = self.data_dir / "uploads" 
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        
        # Create directories
        for directory in [self.data_dir, self.uploads_dir, self.logs_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)
        
        # Database settings
        self.database_path = self.data_dir / "research_agent.db"
        
        # LLM settings - Updated for faster model
        self.default_model = "llama3.2:1b"  # Much faster than llama2:7b
        self.ollama_base_url = "http://localhost:11434"
        self.temperature = 0.7
        self.max_tokens = 512  # Reduced for faster response
        
        # File processing settings
        self.supported_file_types = ['.pdf', '.txt', '.docx']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Search settings
        self.max_search_results = 3  # Reduced for faster processing
        self.similarity_threshold = 0.7
        
        # Load environment variables if available
        self.load_from_env()
    
    def load_from_env(self):
        """Load settings from environment variables"""
        self.default_model = os.getenv('DEFAULT_MODEL', self.default_model)
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', self.ollama_base_url)
        self.temperature = float(os.getenv('TEMPERATURE', self.temperature))
        self.max_tokens = int(os.getenv('MAX_TOKENS', self.max_tokens))
    
    def get_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'uploads_dir': str(self.uploads_dir),
            'database_path': str(self.database_path),
            'default_model': self.default_model,
            'ollama_base_url': self.ollama_base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'supported_file_types': self.supported_file_types,
            'max_file_size': self.max_file_size
        }
    
    def validate_setup(self) -> tuple[bool, list[str]]:
        """Validate that the setup is correct"""
        issues = []
        
        # Check directories
        for name, path in [
            ('data', self.data_dir),
            ('uploads', self.uploads_dir),
            ('logs', self.logs_dir)
        ]:
            if not path.exists():
                issues.append(f"Missing {name} directory: {path}")
            elif not path.is_dir():
                issues.append(f"{name} path is not a directory: {path}")
        
        return len(issues) == 0, issues