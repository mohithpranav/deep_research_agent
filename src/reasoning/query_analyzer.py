"""
Query Analyzer - Analyzes user queries to determine intent and context
"""
from typing import Any, Dict, List, Optional
from enum import Enum
from utils.logger import setup_logger

class QueryIntent(Enum):
    """Enumeration of possible query intents"""
    BIOGRAPHICAL = "biographical"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    CONTACT = "contact"
    GENERAL = "general"

class QueryAnalyzer:
    """Analyzes user queries to understand intent and context"""
    
    def __init__(self):
        self.logger = setup_logger("QueryAnalyzer")
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract intent, keywords, and context"""
        try:
            query_lower = query.lower().strip()
            
            # Determine query type
            query_intent = self._classify_query_intent(query_lower)
            query_type = query_intent.value
            
            # Extract keywords
            keywords = self._extract_keywords(query_lower)
            
            # Determine complexity
            complexity = self._assess_complexity(query_lower)
            
            # Build analysis result
            analysis = {
                'original_query': query,
                'query_type': query_type,
                'query_intent': query_intent,
                'keywords': keywords,
                'complexity': complexity,
                'requires_reasoning': self._requires_reasoning(query_lower),
                'expected_answer_type': self._determine_answer_type(query_type)
            }
            
            self.logger.info(f"Query analysis: {query_type} - {len(keywords)} keywords")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return {
                'original_query': query,
                'query_type': 'general',
                'query_intent': QueryIntent.GENERAL,
                'keywords': query.lower().split(),
                'complexity': 'simple',
                'requires_reasoning': True,
                'expected_answer_type': 'comprehensive'
            }
    
    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Classify the intent of any type of query"""
        
        # Information/factual queries
        if any(phrase in query for phrase in ['what is', 'what are', 'define', 'explain', 'describe']):
            return QueryIntent.GENERAL
        
        # Person/entity queries  
        elif any(phrase in query for phrase in ['who is', 'who are', 'tell me about']):
            return QueryIntent.BIOGRAPHICAL
        
        # Skills/capabilities queries
        elif any(phrase in query for phrase in ['skills', 'abilities', 'capabilities', 'expertise', 'technologies', 'methods']):
            return QueryIntent.SKILLS
        
        # Process/procedure queries
        elif any(phrase in query for phrase in ['how to', 'process', 'procedure', 'steps', 'method']):
            return QueryIntent.EXPERIENCE
        
        # Project/work/research queries
        elif any(phrase in query for phrase in ['projects', 'research', 'studies', 'work', 'findings', 'results']):
            return QueryIntent.PROJECTS
        
        # Learning/education/background queries
        elif any(phrase in query for phrase in ['background', 'history', 'education', 'training', 'learning']):
            return QueryIntent.EDUCATION
        
        # Contact/reference queries
        elif any(phrase in query for phrase in ['contact', 'reference', 'source', 'citation']):
            return QueryIntent.CONTACT
        
        else:
            return QueryIntent.GENERAL
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        
        # Common stop words to filter out
        stop_words = {
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'who',
            'where', 'when', 'why', 'how', 'tell', 'me', 'about'
        }
        
        # Extract words and filter
        words = query.replace('?', '').replace(',', '').split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count <= 3:
            return 'simple'
        elif word_count <= 8:
            return 'moderate'
        else:
            return 'complex'
    
    def _requires_reasoning(self, query: str) -> bool:
        """Determine if query requires AI reasoning vs simple search"""
        
        reasoning_indicators = [
            'explain', 'analyze', 'compare', 'summarize', 'describe',
            'why', 'how', 'what makes', 'tell me about'
        ]
        
        return any(indicator in query for indicator in reasoning_indicators)
    
    def _determine_answer_type(self, query_type: str) -> str:
        """Determine expected answer format"""
        
        answer_types = {
            'biographical': 'comprehensive_profile',
            'skills': 'structured_list',
            'experience': 'chronological_summary',
            'projects': 'project_descriptions',
            'education': 'academic_background',
            'contact': 'contact_info',
            'general': 'comprehensive'
        }
        
        return answer_types.get(query_type, 'comprehensive')

# For backward compatibility
def analyze_query(query: str) -> Dict[str, Any]:
    """Standalone function to analyze queries"""
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)