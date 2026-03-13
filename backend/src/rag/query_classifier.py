from typing import List, Optional, Dict, Any
import re
import logging
from groq import Groq
from src.models.route_identifier import QueryCategory, ClassificationResult

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Intelligent query classifier using Groq API and rule-based fallback."""
    
    def __init__(self, groq_client: Groq):
        self.client = groq_client
        self.classification_prompt = """
You are a query classification expert. Analyze the user's query and classify it into one of these categories:

1. PORTFOLIO: Questions about Prajwal's skills, projects, experience, certifications
2. TECHNICAL: Technical questions about programming, technologies, frameworks
3. GENERAL: General knowledge questions not related to portfolio
4. GREETING: Hello, hi, hey, good morning, etc.
5. GOODBYE: Bye, goodbye, thanks, see you later, etc.
6. UNCLEAR: Ambiguous, unclear, or nonsensical queries
7. OUT_OF_SCOPE: Completely unrelated to portfolio or general conversation

Query: "{query}"

Respond in this exact format:
CATEGORY: [category_name]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of why this category]
KEYWORDS: [comma-separated key terms]

Example response:
CATEGORY: PORTFOLIO
CONFIDENCE: 0.95
REASONING: Directly asks about Prajwal's projects
KEYWORDS: projects, Prajwal, work
"""
        
        # Rule-based patterns for fallback classification
        self.patterns = {
            QueryCategory.GREETING: [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'^\s*$',  # Empty or whitespace
            ],
            QueryCategory.GOODBYE: [
                r'\b(bye|goodbye|thanks|thank you|see you|later)\b',
            ],
            QueryCategory.PORTFOLIO: [
                r'\b(prajwal|portfolio|projects?|skills?|experience|certifications?|resume|cv)\b',
                r'\b(what.*do|tell.*about|show.*work|projects?.*worked)\b',
            ],
            QueryCategory.TECHNICAL: [
                r'\b(python|javascript|react|nodejs|database|api|framework|library)\b',
                r'\b(how.*implement|what.*technology|best.*practice|algorithm)\b',
            ]
        }
    
    def classify_with_groq(self, query: str) -> Optional[ClassificationResult]:
        """Use Groq API to classify the query."""
        try:
            prompt = self.classification_prompt.format(query=query)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a query classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=150
            )
            
            classification_text = response.choices[0].message.content.strip()
            
            # Parse the response
            category = None
            confidence = 0.0
            reasoning = ""
            keywords = []
            
            for line in classification_text.split('\n'):
                line = line.strip()
                if line.startswith('CATEGORY:'):
                    category_name = line.split(':', 1)[1].strip()
                    try:
                        category = QueryCategory[category_name.upper()]
                    except KeyError:
                        logger.warning(f"Unknown category: {category_name}")
                        continue
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        logger.warning(f"Invalid confidence value: {line}")
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif line.startswith('KEYWORDS:'):
                    keywords_str = line.split(':', 1)[1].strip()
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
            if category and confidence > 0.3:  # Minimum confidence threshold
                return ClassificationResult(
                    category=category,
                    confidence=confidence,
                    reasoning=reasoning,
                    keywords=keywords
                )
            
        except Exception as e:
            logger.error(f"Groq classification failed: {e}")
        
        return None
    
    def classify_with_rules(self, query: str) -> ClassificationResult:
        """Use rule-based patterns for classification."""
        query_lower = query.lower().strip()
        
        # Check for empty or very short queries
        if not query or len(query.strip()) < 2:
            return ClassificationResult(
                category=QueryCategory.UNCLEAR,
                confidence=0.9,
                reasoning="Query is too short or empty",
                keywords=[]
            )
        
        # Apply patterns
        category_scores = {}
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    category_scores[category] = category_scores.get(category, 0) + 1
        
        # Determine best match
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = min(0.8, 0.4 + (category_scores[best_category] * 0.2))
            
            return ClassificationResult(
                category=best_category,
                confidence=confidence,
                reasoning=f"Matched {category_scores[best_category]} pattern(s)",
                keywords=list(set(re.findall(r'\b\w+\b', query_lower)))
            )
        
        # Default to general if no patterns match
        return ClassificationResult(
            category=QueryCategory.GENERAL,
            confidence=0.3,
            reasoning="No specific patterns matched",
            keywords=list(set(re.findall(r'\b\w+\b', query_lower)))
        )
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query using Groq API first, fallback to rules if needed.
        
        Args:
            query: The user's input query
            
        Returns:
            ClassificationResult with category, confidence, and reasoning
        """
        if not query or not query.strip():
            return ClassificationResult(
                category=QueryCategory.UNCLEAR,
                confidence=1.0,
                reasoning="Empty query",
                keywords=[]
            )
        
        # Try Groq classification first
        groq_result = self.classify_with_groq(query)
        if groq_result and groq_result.confidence >= 0.7:
            logger.info(f"Groq classified query as {groq_result.category.value} with confidence {groq_result.confidence}")
            return groq_result
        
        # Fallback to rule-based classification
        rule_result = self.classify_with_rules(query)
        
        # If Groq had low confidence but gave a result, prefer it over rules
        if groq_result and groq_result.confidence >= 0.4:
            logger.info(f"Using Groq classification with low confidence: {groq_result.category.value}")
            return groq_result
        
        logger.info(f"Using rule-based classification: {rule_result.category.value}")
        return rule_result