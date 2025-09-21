"""
Response Synthesizer - Creates comprehensive responses from multiple sources
"""
from typing import Any, Dict, List, Optional
from utils.logger import setup_logger
from .llm_client import LLMClient

class ResponseSynthesizer:
    """Synthesizes comprehensive responses from search results and context"""
    
    def __init__(self):
        self.logger = setup_logger("ResponseSynthesizer")
        self.llm_client = LLMClient()
    
    def synthesize(self, query: str, context: List[str], 
                   query_info: Dict[str, Any] = None) -> str:
        """Synthesize a comprehensive response using AI reasoning"""
        try:
            print(f"🔬 SYNTHESIZING RESPONSE for: '{query}'")
            print(f"📚 Using {len(context)} context pieces")
            
            # Check if AI is available
            if not self.llm_client.is_healthy():
                print("❌ LLM not healthy, falling back to basic response")
                return self._fallback_response(query, context)
            
            # Filter and clean context
            valid_context = self._prepare_context(context)
            
            if not valid_context:
                return self._handle_no_context(query)
            
            # Determine response strategy based on query type
            query_type = query_info.get('query_type', 'general') if query_info else 'general'
            system_prompt = self._get_system_prompt(query_type)
            
            print(f"🎯 QUERY TYPE: {query_type}")
            print(f"🧠 SENDING TO AI for comprehensive analysis...")
            
            # Generate AI response
            ai_response = self.llm_client.generate_response(
                prompt=query,
                context=valid_context,
                system_prompt=system_prompt
            )
            
            if ai_response and len(ai_response.strip()) > 50:
                print(f"✅ AI SYNTHESIS COMPLETE: {len(ai_response)} characters")
                return self._format_response(ai_response, query_type)
            else:
                print("⚠️ AI response too short, using fallback")
                return self._fallback_response(query, valid_context)
                
        except Exception as e:
            self.logger.error(f"Response synthesis failed: {e}")
            print(f"❌ SYNTHESIS ERROR: {e}")
            return self._fallback_response(query, context)
    
    def _prepare_context(self, context: List[str]) -> List[str]:
        """Clean and prepare context for AI processing"""
        valid_context = []
        
        for item in context:
            if item and isinstance(item, str):
                # Clean up formatting
                cleaned = item.replace('📄 **', '').replace('**:', ':')
                cleaned = cleaned.replace('📄 ', '').strip()
                
                # Only include substantial content
                if len(cleaned) > 20:
                    valid_context.append(cleaned)
        
        return valid_context[:5]  # Limit to top 5 most relevant
    
    def _get_system_prompt(self, query_type: str) -> str:
        """Get specialized system prompt for generic document analysis"""
        
        base_instructions = """You are a Deep Researcher Agent analyzing uploaded documents. Your task is to provide comprehensive, insightful analysis of any type of content - whether it's research papers, reports, articles, books, personal documents, technical documentation, or any other text."""
        
        type_specific_prompts = {
            'biographical': f"""{base_instructions}

For questions about people, entities, or subjects mentioned in the documents:
- Provide comprehensive information about the person/entity
- Include background, roles, activities, and key details from the documents
- Structure information logically (Background, Current Status, Key Activities, Notable Points)
- Use specific details and evidence from the document content""",
            
            'skills': f"""{base_instructions}

For questions about skills, technologies, methods, or capabilities:
- Identify and categorize different types of skills/technologies/methods mentioned
- Organize by category (Technical, Methodological, Tools, Approaches, etc.)
- Highlight specific details and proficiency levels where mentioned
- Include context about how these are used or applied""",
            
            'experience': f"""{base_instructions}

For questions about processes, procedures, experiences, or activities:
- Describe processes and activities mentioned in the documents
- Include specific steps, methods, and approaches
- Highlight outcomes, results, and impacts where mentioned
- Organize information chronologically or by importance""",
            
            'projects': f"""{base_instructions}

For questions about projects, research, studies, or work:
- Describe each project/study/work with clear objectives and scope
- Include methodologies, approaches, and tools used
- Mention outcomes, findings, and results
- Organize by relevance, timeline, or significance""",
            
            'education': f"""{base_instructions}

For questions about background, education, history, or foundational information:
- Include historical context and background information
- Mention educational elements, training, or development aspects
- Include foundational concepts and their development
- Structure chronologically or thematically""",
            
            'contact': f"""{base_instructions}

For questions about sources, references, contacts, or citations:
- Provide available contact information, references, or sources
- Include links, citations, or reference materials mentioned
- Format information clearly and accessibly""",
            
            'general': f"""{base_instructions}

For general questions:
- Provide comprehensive analysis of the relevant document content
- Think deeply about the information and its implications
- Structure response logically with clear sections
- Use specific evidence and details from the documents
- Provide insights and analysis beyond just summarizing facts"""
        }
        
        return type_specific_prompts.get(query_type, type_specific_prompts['general'])
    
    def _format_response(self, ai_response: str, query_type: str) -> str:
        """Format the AI response for any document type"""
        
        # Clean up the response
        formatted_response = ai_response.strip()
        
        # Generic headers
        headers = {
            'biographical': "## Entity/Subject Analysis\n\n",
            'skills': "## Skills & Capabilities Analysis\n\n", 
            'experience': "## Process & Activity Analysis\n\n",
            'projects': "## Project & Research Analysis\n\n",
            'education': "## Background & Context Analysis\n\n",
            'contact': "## Reference & Source Information\n\n",
            'general': "## Document Analysis Results\n\n"
        }
        
        header = headers.get(query_type, "## Deep Research Analysis\n\n")
        
        # Generic AI indicator
        footer = "\n\n*🔬 Deep analysis generated by AI research agent*"
        
        return header + formatted_response + footer
    
    def _handle_no_context(self, query: str) -> str:
        """Handle queries when no relevant context is found"""
        return f"""## No Relevant Information Found

I don't have enough relevant information in the uploaded documents to answer your question about "{query}". 

**To get better results:**
1. Make sure you've uploaded relevant documents
2. Check that the documents contain information related to your question  
3. Try rephrasing your question with different keywords

If you've uploaded documents but I still can't find relevant information, the documents might not contain the specific details you're looking for."""
    
    def _fallback_response(self, query: str, context: List[str]) -> str:
        """Provide fallback response when AI is unavailable"""
        if context:
            response = f"## Document Search Results\n\n*Based on your uploaded documents, here's what I found regarding '{query}':*\n\n"
            
            for i, snippet in enumerate(context[:3], 1):
                if snippet.strip():
                    clean_snippet = snippet.replace('📄 **', '').replace('**:', ':')
                    response += f"**Finding {i}:**\n{clean_snippet}\n\n"
            
            response += "\n⚠️ *Note: AI analysis unavailable. For comprehensive AI-powered responses, ensure Ollama is running properly.*"
            return response
        else:
            return self._handle_no_context(query)
    
    def is_healthy(self) -> bool:
        """Check if the synthesizer can function properly"""
        return self.llm_client.is_healthy()