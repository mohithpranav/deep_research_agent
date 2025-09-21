"""
Multi-Step Reasoner - Orchestrates complex reasoning using retrieved context and LLM
"""
from typing import Any, Dict, List, Optional
from utils.logger import setup_logger
from .query_analyzer import QueryAnalyzer, QueryIntent
from .response_synthesizer import ResponseSynthesizer
from .llm_client import LLMClient


class MultiStepReasoner:
    """Handles complex multi-step reasoning queries"""
    
    def __init__(self):
        self.logger = setup_logger("MultiStepReasoner")
        self.query_analyzer = QueryAnalyzer()
        self.synthesizer = ResponseSynthesizer()
        self.llm_client = LLMClient()
    
    def reason(self, query: str, context: List[str]) -> str:
        """Perform multi-step reasoning on complex queries"""
        try:
            # Analyze the query first
            query_info = self.query_analyzer.analyze(query)
            
            if query_info['complexity'] == 'complex':
                return self._handle_complex_query(query, context, query_info)
            else:
                # Use regular synthesis for simple queries
                return self.synthesizer.synthesize(query, context, query_info)
                
        except Exception as e:
            self.logger.error(f"Multi-step reasoning failed: {e}")
            return self.synthesizer.synthesize(query, context)
    
    def _handle_complex_query(self, query: str, context: List[str], 
                            query_info: Dict[str, Any]) -> str:
        """Handle complex queries requiring multi-step reasoning"""
        
        # Break down complex query into sub-questions
        sub_questions = self._decompose_query(query, query_info)
        
        # Process each sub-question
        partial_answers = []
        for sub_q in sub_questions:
            answer = self.synthesizer.synthesize(sub_q, context)
            partial_answers.append(answer)
        
        # Combine answers into comprehensive response
        return self._combine_answers(query, partial_answers)
    
    def _decompose_query(self, query: str, query_info: Dict[str, Any]) -> List[str]:
        """Break down complex query into simpler sub-questions"""
        
        query_type = query_info.get('query_type', 'general')
        
        # Define decomposition strategies
        decomposition_map = {
            'biographical': [
                "What is the person's background and current role?",
                "What are their key skills and expertise?", 
                "What notable projects have they worked on?"
            ],
            'general': [query]  # Fallback to original query
        }
        
        return decomposition_map.get(query_type, [query])
    
    def _combine_answers(self, original_query: str, partial_answers: List[str]) -> str:
        """Combine partial answers into a comprehensive response"""
        
        combined = f"## Comprehensive Analysis: {original_query}\n\n"
        
        for i, answer in enumerate(partial_answers, 1):
            if answer and len(answer.strip()) > 20:
                combined += f"### Aspect {i}\n{answer}\n\n"
        
        return combined
    
    def is_healthy(self) -> bool:
        """Check if the reasoner can function properly"""
        return self.llm_client.is_healthy()