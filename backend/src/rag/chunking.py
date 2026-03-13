import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import nltk
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
import logging

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    CODE = "code"
    MARKDOWN = "markdown"

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]
    chunk_id: str
    word_count: int
    char_count: int

class IntelligentChunker:
    """Intelligent document chunking with multiple strategies"""
    
    def __init__(self, 
                 default_chunk_size: int = 1000,
                 default_chunk_overlap: int = 200,
                 default_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.default_strategy = default_strategy
        
        # Initialize NLTK data (download if not present)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK punkt: {e}")
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK punkt_tab: {e}")
    
    def chunk_document(self, 
                      content: str,
                      doc_type: str = "text",
                      strategy: Optional[ChunkingStrategy] = None,
                      chunk_size: Optional[int] = None,
                      chunk_overlap: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk document using specified strategy
        
        Args:
            content: Document content to chunk
            doc_type: Type of document (text, pdf, json, csv, etc.)
            strategy: Chunking strategy to use
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            metadata: Additional metadata to attach to chunks
        
        Returns:
            List of chunks with metadata
        """
        if not content or not content.strip():
            return []
        
        strategy = strategy or self._select_strategy(doc_type)
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        metadata = metadata or {}
        
        logger.info(f"Chunking document with strategy: {strategy.value}, size: {chunk_size}, overlap: {chunk_overlap}")
        
        if strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(content, chunk_size, chunk_overlap, metadata)
        elif strategy == ChunkingStrategy.CHARACTER:
            return self._chunk_character(content, chunk_size, chunk_overlap, metadata)
        elif strategy == ChunkingStrategy.TOKEN:
            return self._chunk_token(content, chunk_size, chunk_overlap, metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(content, chunk_size, chunk_overlap, metadata)
        elif strategy == ChunkingStrategy.STRUCTURAL:
            return self._chunk_structural(content, doc_type, metadata)
        elif strategy == ChunkingStrategy.CODE:
            return self._chunk_code(content, chunk_size, chunk_overlap, metadata)
        elif strategy == ChunkingStrategy.MARKDOWN:
            return self._chunk_markdown(content, chunk_size, chunk_overlap, metadata)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def _select_strategy(self, doc_type: str) -> ChunkingStrategy:
        """Select optimal chunking strategy based on document type"""
        strategy_map = {
            'pdf': ChunkingStrategy.RECURSIVE,
            'text': ChunkingStrategy.RECURSIVE,
            'json': ChunkingStrategy.STRUCTURAL,
            'csv': ChunkingStrategy.STRUCTURAL,
            'excel': ChunkingStrategy.STRUCTURAL,
            'xml': ChunkingStrategy.STRUCTURAL,
            'html': ChunkingStrategy.STRUCTURAL,
            'web_html': ChunkingStrategy.STRUCTURAL,
            'markdown': ChunkingStrategy.MARKDOWN,
            'code': ChunkingStrategy.CODE,
        }
        
        return strategy_map.get(doc_type, self.default_strategy)
    
    def _chunk_recursive(self, content: str, chunk_size: int, chunk_overlap: int, 
                        metadata: Dict[str, Any]) -> List[Chunk]:
        """Recursive character-based chunking"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        texts = splitter.split_text(content)
        return self._create_chunks(texts, "recursive", metadata)
    
    def _chunk_character(self, content: str, chunk_size: int, chunk_overlap: int,
                        metadata: Dict[str, Any]) -> List[Chunk]:
        """Simple character-based chunking"""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        texts = splitter.split_text(content)
        return self._create_chunks(texts, "character", metadata)
    
    def _chunk_token(self, content: str, chunk_size: int, chunk_overlap: int,
                    metadata: Dict[str, Any]) -> List[Chunk]:
        """Token-based chunking"""
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        texts = splitter.split_text(content)
        return self._create_chunks(texts, "token", metadata)
    
    def _chunk_semantic(self, content: str, chunk_size: int, chunk_overlap: int,
                       metadata: Dict[str, Any]) -> List[Chunk]:
        """Semantic chunking based on sentence boundaries"""
        try:
            sentences = nltk.sent_tokenize(content)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # Start new chunk if current one is too large
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep overlap
                overlap_sentences = []
                overlap_length = 0
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return self._create_chunks(chunks, "semantic", metadata)
    
    def _chunk_structural(self, content: str, doc_type: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Structure-aware chunking for structured documents"""
        if doc_type == "json":
            return self._chunk_json_structural(content, metadata)
        elif doc_type in ["csv", "excel"]:
            return self._chunk_tabular_structural(content, metadata)
        elif doc_type in ["html", "xml", "web_html"]:
            return self._chunk_html_structural(content, metadata)
        else:
            # Fallback to recursive for unknown types
            return self._chunk_recursive(content, self.default_chunk_size, 
                                       self.default_chunk_overlap, metadata)
    
    def _chunk_json_structural(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk JSON documents preserving structure"""
        try:
            import json
            data = json.loads(content)
            
            chunks = []
            
            if isinstance(data, list):
                # Group array items into chunks
                items_per_chunk = max(1, len(data) // 10)  # Rough heuristic
                
                for i in range(0, len(data), items_per_chunk):
                    chunk_data = data[i:i + items_per_chunk]
                    chunk_content = json.dumps(chunk_data, indent=2)
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'item_range': f"{i}-{min(i + items_per_chunk - 1, len(data) - 1)}",
                        'item_count': len(chunk_data)
                    })
                    
                    chunks.append(chunk_content)
            else:
                # For objects, create single chunk
                chunks.append(content)
            
            return self._create_chunks(chunks, "json_structural", metadata)
            
        except json.JSONDecodeError:
            # Fallback to recursive if JSON parsing fails
            return self._chunk_recursive(content, self.default_chunk_size,
                                       self.default_chunk_overlap, metadata)
    
    def _chunk_tabular_structural(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk tabular data preserving row structure"""
        lines = content.strip().split('\n')
        
        if len(lines) <= 1:
            return self._create_chunks([content], "tabular_structural", metadata)
        
        # Assume first line is header
        header = lines[0]
        data_lines = lines[1:]
        
        # Group rows into chunks
        rows_per_chunk = max(1, len(data_lines) // 5)  # Rough heuristic
        chunks = []
        
        for i in range(0, len(data_lines), rows_per_chunk):
            chunk_lines = [header] + data_lines[i:i + rows_per_chunk]
            chunk_content = '\n'.join(chunk_lines)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'row_range': f"{i + 1}-{min(i + rows_per_chunk, len(data_lines))}",
                'row_count': len(data_lines[i:i + rows_per_chunk])
            })
            
            chunks.append(chunk_content)
        
        return self._create_chunks(chunks, "tabular_structural", metadata)
    
    def _chunk_html_structural(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk HTML documents preserving structure"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text by sections
            sections = []
            
            # Get title
            title = soup.find('title')
            if title:
                sections.append(f"TITLE: {title.get_text().strip()}")
            
            # Get headings and their content
            for heading_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headings = soup.find_all(heading_tag)
                for heading in headings:
                    heading_text = heading.get_text().strip()
                    
                    # Get content until next heading
                    content_parts = []
                    current = heading.find_next_sibling()
                    
                    while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if current.name == 'p':
                            content_parts.append(current.get_text().strip())
                        elif current.name in ['ul', 'ol']:
                            items = current.find_all('li')
                            for item in items:
                                content_parts.append(f"- {item.get_text().strip()}")
                        
                        current = current.find_next_sibling()
                    
                    if heading_text or content_parts:
                        section_content = f"{heading_text}\n" + "\n".join(content_parts)
                        sections.append(section_content)
            
            # If no structure found, fall back to recursive
            if not sections:
                text = soup.get_text()
                return self._chunk_recursive(text, self.default_chunk_size,
                                           self.default_chunk_overlap, metadata)
            
            return self._create_chunks(sections, "html_structural", metadata)
            
        except ImportError:
            logger.warning("beautifulsoup4 not available for HTML structural chunking")
            # Fallback to recursive
            return self._chunk_recursive(content, self.default_chunk_size,
                                       self.default_chunk_overlap, metadata)
        except Exception as e:
            logger.error(f"Error in HTML structural chunking: {e}")
            # Fallback to recursive
            return self._chunk_recursive(content, self.default_chunk_size,
                                       self.default_chunk_overlap, metadata)
    
    def _chunk_code(self, content: str, chunk_size: int, chunk_overlap: int,
                   metadata: Dict[str, Any]) -> List[Chunk]:
        """Code-aware chunking"""
        splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        texts = splitter.split_text(content)
        return self._create_chunks(texts, "code", metadata)
    
    def _chunk_markdown(self, content: str, chunk_size: int, chunk_overlap: int,
                       metadata: Dict[str, Any]) -> List[Chunk]:
        """Markdown-aware chunking"""
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        texts = splitter.split_text(content)
        return self._create_chunks(texts, "markdown", metadata)
    
    def _create_chunks(self, texts: List[str], strategy: str, 
                      metadata: Dict[str, Any]) -> List[Chunk]:
        """Create Chunk objects from text list"""
        chunks = []
        current_index = 0
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            
            text = text.strip()
            start_index = current_index
            end_index = current_index + len(text)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_strategy': strategy,
                'char_count': len(text),
                'word_count': len(text.split()),
            })
            
            chunk = Chunk(
                content=text,
                start_index=start_index,
                end_index=end_index,
                metadata=chunk_metadata,
                chunk_id=f"chunk_{i}_{hash(text) % 10000}",
                word_count=len(text.split()),
                char_count=len(text)
            )
            
            chunks.append(chunk)
            current_index = end_index
        
        return chunks
    
    def analyze_content(self, content: str, doc_type: str = "text") -> Dict[str, Any]:
        """Analyze content to recommend chunking parameters"""
        if not content:
            return {}
        
        analysis = {
            'total_chars': len(content),
            'total_words': len(content.split()),
            'total_lines': len(content.split('\n')),
            'doc_type': doc_type,
            'recommended_strategy': self._select_strategy(doc_type).value,
        }
        
        # Estimate optimal chunk size
        if analysis['total_words'] < 100:
            analysis['recommended_chunk_size'] = 50
            analysis['recommended_chunk_overlap'] = 10
        elif analysis['total_words'] < 500:
            analysis['recommended_chunk_size'] = 200
            analysis['recommended_chunk_overlap'] = 50
        elif analysis['total_words'] < 2000:
            analysis['recommended_chunk_size'] = 500
            analysis['recommended_chunk_overlap'] = 100
        else:
            analysis['recommended_chunk_size'] = 1000
            analysis['recommended_chunk_overlap'] = 200
        
        # Calculate estimated chunks
        chunk_size = analysis['recommended_chunk_size']
        overlap = analysis['recommended_chunk_overlap']
        
        if chunk_size > overlap:
            effective_size = chunk_size - overlap
            estimated_chunks = max(1, (analysis['total_words'] - overlap) / effective_size)
            analysis['estimated_chunks'] = int(estimated_chunks) + 1
        else:
            analysis['estimated_chunks'] = 1
        
        return analysis
    
    def optimize_chunks(self, chunks: List[Chunk], target_size: int = None) -> List[Chunk]:
        """Optimize chunk sizes by merging small chunks or splitting large ones"""
        if not chunks:
            return []
        
        target_size = target_size or self.default_chunk_size
        optimized_chunks = []
        
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue
            
            # If current chunk is too small, try to merge
            if current_chunk.char_count < target_size * 0.5:
                # Try to merge with next chunk
                merged_content = current_chunk.content + " " + chunk.content
                merged_metadata = current_chunk.metadata.copy()
                merged_metadata['merged_chunks'] = merged_metadata.get('merged_chunks', 0) + 1
                
                # Create new merged chunk
                merged_chunk = Chunk(
                    content=merged_content,
                    start_index=current_chunk.start_index,
                    end_index=chunk.end_index,
                    metadata=merged_metadata,
                    chunk_id=f"merged_{current_chunk.chunk_id}_{chunk.chunk_id}",
                    word_count=len(merged_content.split()),
                    char_count=len(merged_content)
                )
                
                # If merged chunk is not too large, use it
                if merged_chunk.char_count <= target_size * 1.5:
                    current_chunk = merged_chunk
                else:
                    # Otherwise, keep current and start new
                    optimized_chunks.append(current_chunk)
                    current_chunk = chunk
            else:
                # Current chunk is good size, add it and start new
                optimized_chunks.append(current_chunk)
                current_chunk = chunk
        
        # Add final chunk
        if current_chunk:
            optimized_chunks.append(current_chunk)
        
        return optimized_chunks