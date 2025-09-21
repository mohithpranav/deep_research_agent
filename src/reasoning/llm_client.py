"""
LLM Client for interacting with Ollama
"""
import requests
import json
from typing import Dict, List, Optional, Any
from config.settings import Settings
from utils.logger import setup_logger

class LLMClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = setup_logger("LLMClient")
        self.base_url = self.settings.ollama_base_url
        self.model = self.settings.default_model
        
    def is_healthy(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except Exception as e:
            self.logger.error(f"Failed to get models: {e}")
            return []
    
    def generate_response(self, prompt: str, context: List[str] = None, 
                         system_prompt: str = None) -> str:
        """Generate response using Ollama with optimized settings"""
        try:
            # Build optimized prompt
            full_prompt = self._build_optimized_prompt(prompt, context, system_prompt)
            
            print(f"🧠 SENDING TO LLM: {self.model}")
            print(f"📝 PROMPT LENGTH: {len(full_prompt)} characters")
            
            # Prepare optimized request
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.settings.temperature,
                    "num_predict": self.settings.max_tokens,
                    "top_k": 40,  # Limit choices for faster response
                    "top_p": 0.9,  # Focus on likely tokens
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n\n"]  # Stop at natural breaks
                }
            }
            
            # Try with progressively longer timeouts
            timeouts = [15, 30, 45]  # Start with shorter timeout
            
            for timeout in timeouts:
                try:
                    print(f"⏱️ Trying with {timeout}s timeout...")
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result.get('response', '').strip()
                        
                        if generated_text and len(generated_text) > 10:
                            print(f"✅ LLM RESPONSE: {len(generated_text)} characters")
                            self.logger.info(f"Generated response: {len(generated_text)} chars")
                            return generated_text
                        else:
                            print("⚠️ Empty response, trying longer timeout...")
                            continue
                    else:
                        print(f"❌ HTTP {response.status_code}, trying longer timeout...")
                        continue
                        
                except requests.exceptions.Timeout:
                    print(f"⏱️ {timeout}s timeout exceeded, trying longer...")
                    continue
                except Exception as e:
                    print(f"❌ Request failed: {e}")
                    break
            
            # If all timeouts failed
            return self._get_fallback_response(prompt, context)
            
        except Exception as e:
            error_msg = f"LLM generation failed: {e}"
            print(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return self._get_fallback_response(prompt, context)

    def _build_optimized_prompt(self, query: str, context: List[str] = None, 
                               system_prompt: str = None) -> str:
        """Build generic prompt for any document analysis"""
        
        # Generic document analysis prompt
        default_system = """You are a Deep Researcher Agent - an advanced document analysis AI. Your job is to thoroughly analyze any uploaded document content and provide comprehensive, insightful responses.

CAPABILITIES:
- Analyze any type of document (research papers, reports, books, articles, personal documents, etc.)
- Extract key information and insights
- Answer questions based on document content
- Provide detailed analysis and reasoning
- Synthesize information from multiple sources

APPROACH:
- Read and understand the provided document content carefully
- Think deeply about the information presented
- Provide thorough, well-structured responses
- Use specific details from the documents to support your answers
- If information is not available in the documents, clearly state this"""
        
        system = system_prompt or default_system
        
        # Build generic context section
        context_section = ""
        if context and any(context):
            context_section = f"\n=== DOCUMENT CONTENT FOR ANALYSIS ===\n"
            
            for i, doc_snippet in enumerate(context[:3], 1):
                if doc_snippet.strip():
                    clean_snippet = doc_snippet.replace('📄 **', '').replace('**:', ':')[:400]
                    context_section += f"Document Extract {i}:\n{clean_snippet}\n\n"
            
            context_section += "=== END DOCUMENT CONTENT ===\n\n"
        
        # Generic analysis prompt
        full_prompt = f"""{system}

{context_section}Research Query: {query}

Please analyze the document content above and provide a comprehensive response to the query. Think deeply about the information and provide detailed insights:

Analysis:"""
        
        return full_prompt

    def _get_fallback_response(self, prompt: str, context: List[str]) -> str:
        """Provide fallback response when LLM fails"""
        return f"I found information about '{prompt}' but the AI model is currently slow. Here's a quick summary based on your documents: " + (context[0][:200] + "..." if context else "No specific details found.")