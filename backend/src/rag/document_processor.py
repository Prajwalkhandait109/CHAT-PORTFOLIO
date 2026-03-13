import os
import json
import csv
import re
import requests
import PyPDF2
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Enhanced document class supporting multiple formats"""
    content: str
    metadata: Dict[str, Any]
    doc_type: str
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    file_hash: Optional[str] = None

class DocumentProcessor:
    """Multi-format document processor for various file types and web content"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._process_text,
            '.pdf': self._process_pdf,
            '.json': self._process_json,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.md': self._process_markdown,
            '.html': self._process_html,
            '.xml': self._process_xml,
        }
        
        self.web_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Process a file and return list of documents"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        processor = self.supported_extensions[extension]
        
        try:
            documents = processor(file_path)
            logger.info(f"Successfully processed {len(documents)} documents from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def process_url(self, url: str, include_metadata: bool = True) -> List[Document]:
        """Process web content and return documents"""
        try:
            response = requests.get(url, headers=self.web_headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type:
                return self._process_pdf_from_bytes(response.content, url)
            elif 'text/html' in content_type:
                return self._process_web_html(response.text, url, include_metadata)
            elif 'application/json' in content_type:
                return self._process_json_from_string(response.text, url)
            else:
                # Default to text processing
                return self._process_web_text(response.text, url, include_metadata)
                
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            raise
    
    def process_text_content(self, text: str, source: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process raw text content"""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'source': source,
            'doc_type': 'text',
            'processing_timestamp': datetime.now().isoformat(),
            'char_count': len(text),
            'word_count': len(text.split()),
        })
        
        doc = Document(
            content=text.strip(),
            metadata=metadata,
            doc_type='text',
            source=source
        )
        
        return [doc]
    
    def _process_text(self, file_path: Path) -> List[Document]:
        """Process plain text files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        
        return self.process_text_content(content, str(file_path), metadata)
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF files"""
        documents = []
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            metadata = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'page_count': len(pdf_reader.pages),
                'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            }
            
            # Try to extract PDF metadata
            if pdf_reader.metadata:
                pdf_meta = pdf_reader.metadata
                metadata.update({
                    'title': pdf_meta.get('/Title', ''),
                    'author': pdf_meta.get('/Author', ''),
                    'subject': pdf_meta.get('/Subject', ''),
                    'creator': pdf_meta.get('/Creator', ''),
                    'producer': pdf_meta.get('/Producer', ''),
                })
            
            # Process each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        page_metadata = metadata.copy()
                        page_metadata.update({
                            'page_number': page_num + 1,
                            'doc_type': 'pdf',
                            'source': f"{file_path}#page={page_num + 1}"
                        })
                        
                        doc = Document(
                            content=text.strip(),
                            metadata=page_metadata,
                            doc_type='pdf',
                            source=str(file_path),
                            title=metadata.get('title'),
                            author=metadata.get('author'),
                            page_count=len(pdf_reader.pages)
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
        
        return documents
    
    def _process_pdf_from_bytes(self, pdf_bytes: bytes, source: str) -> List[Document]:
        """Process PDF from bytes (for web downloads)"""
        from io import BytesIO
        
        documents = []
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        
        metadata = {
            'page_count': len(pdf_reader.pages),
            'source_url': source,
            'doc_type': 'pdf',
        }
        
        # Process each page
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    page_metadata = metadata.copy()
                    page_metadata['page_number'] = page_num + 1
                    
                    doc = Document(
                        content=text.strip(),
                        metadata=page_metadata,
                        doc_type='pdf',
                        source=source,
                        page_count=len(pdf_reader.pages)
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        return documents
    
    def _process_json(self, file_path: Path) -> List[Document]:
        """Process JSON files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._process_json_data(data, str(file_path))
    
    def _process_json_from_string(self, json_str: str, source: str) -> List[Document]:
        """Process JSON from string"""
        data = json.loads(json_str)
        return self._process_json_data(data, source)
    
    def _process_json_data(self, data: Any, source: str) -> List[Document]:
        """Process JSON data and convert to documents"""
        documents = []
        
        def flatten_json(obj: Any, prefix: str = '') -> Dict[str, Any]:
            """Flatten nested JSON structure"""
            result = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        result.update(flatten_json(value, new_key))
                    else:
                        result[new_key] = str(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}[{i}]"
                    if isinstance(item, (dict, list)):
                        result.update(flatten_json(item, new_key))
                    else:
                        result[new_key] = str(item)
            else:
                result[prefix] = str(obj)
            return result
        
        if isinstance(data, list):
            # Process array of objects
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    flattened = flatten_json(item)
                    content_parts = []
                    for key, value in flattened.items():
                        content_parts.append(f"{key}: {value}")
                    
                    content = "\n".join(content_parts)
                    
                    metadata = {
                        'item_index': i,
                        'item_type': type(item).__name__,
                        'source': source,
                        'doc_type': 'json',
                        'field_count': len(flattened)
                    }
                    
                    doc = Document(
                        content=content,
                        metadata=metadata,
                        doc_type='json',
                        source=source
                    )
                    documents.append(doc)
        else:
            # Process single object or primitive
            flattened = flatten_json(data)
            content_parts = []
            for key, value in flattened.items():
                content_parts.append(f"{key}: {value}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'source': source,
                'doc_type': 'json',
                'field_count': len(flattened),
                'root_type': type(data).__name__
            }
            
            doc = Document(
                content=content,
                metadata=metadata,
                doc_type='json',
                source=source
            )
            documents.append(doc)
        
        return documents
    
    def _process_csv(self, file_path: Path) -> List[Document]:
        """Process CSV files"""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        metadata = {
            'file_name': file_path.name,
            'row_count': len(rows),
            'column_count': len(reader.fieldnames) if reader.fieldnames else 0,
            'columns': reader.fieldnames,
            'source': str(file_path),
            'doc_type': 'csv'
        }
        
        # Process each row
        for i, row in enumerate(rows):
            content_parts = []
            for key, value in row.items():
                if value and str(value).strip():
                    content_parts.append(f"{key}: {value}")
            
            content = "\n".join(content_parts)
            
            row_metadata = metadata.copy()
            row_metadata['row_index'] = i
            
            doc = Document(
                content=content,
                metadata=row_metadata,
                doc_type='csv',
                source=str(file_path)
            )
            documents.append(doc)
        
        return documents
    
    def _process_excel(self, file_path: Path) -> List[Document]:
        """Process Excel files"""
        documents = []
        
        try:
            # Read all sheets
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, df in excel_data.items():
                # Convert DataFrame to records
                records = df.to_dict('records')
                
                metadata = {
                    'file_name': file_path.name,
                    'sheet_name': sheet_name,
                    'row_count': len(records),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'source': str(file_path),
                    'doc_type': 'excel'
                }
                
                # Process each row
                for i, record in enumerate(records):
                    content_parts = []
                    for key, value in record.items():
                        if pd.notna(value) and str(value).strip():
                            content_parts.append(f"{key}: {value}")
                    
                    content = "\n".join(content_parts)
                    
                    row_metadata = metadata.copy()
                    row_metadata['row_index'] = i
                    
                    doc = Document(
                        content=content,
                        metadata=row_metadata,
                        doc_type='excel',
                        source=str(file_path)
                    )
                    documents.append(doc)
                    
        except ImportError:
            logger.error("pandas and openpyxl required for Excel processing. Install with: pip install pandas openpyxl")
            raise
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            raise
        
        return documents
    
    def _process_markdown(self, file_path: Path) -> List[Document]:
        """Process Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        
        return self.process_text_content(content, str(file_path), metadata)
    
    def _process_html(self, file_path: Path) -> List[Document]:
        """Process HTML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return self._process_html_content(html_content, str(file_path))
    
    def _process_xml(self, file_path: Path) -> List[Document]:
        """Process XML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        return self._process_xml_content(xml_content, str(file_path))
    
    def _process_web_html(self, html_content: str, url: str, include_metadata: bool = True) -> List[Document]:
        """Process HTML content from web"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            metadata = {
                'source_url': url,
                'doc_type': 'web_html',
                'processing_timestamp': datetime.now().isoformat(),
            }
            
            if include_metadata:
                # Extract title
                title = soup.find('title')
                if title:
                    metadata['title'] = title.get_text().strip()
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc:
                    metadata['description'] = meta_desc.get('content', '')
                
                # Extract headings
                headings = []
                for heading_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    for heading in soup.find_all(heading_tag):
                        headings.append(heading.get_text().strip())
                if headings:
                    metadata['headings'] = headings[:10]  # Limit to first 10
            
            return self.process_text_content(text, url, metadata)
            
        except ImportError:
            logger.error("beautifulsoup4 required for HTML processing. Install with: pip install beautifulsoup4")
            raise
        except Exception as e:
            logger.error(f"Error processing HTML content: {str(e)}")
            raise
    
    def _process_web_text(self, text_content: str, url: str, include_metadata: bool = True) -> List[Document]:
        """Process plain text content from web"""
        metadata = {
            'source_url': url,
            'doc_type': 'web_text',
            'processing_timestamp': datetime.now().isoformat(),
        }
        
        return self.process_text_content(text_content, url, metadata)
    
    def _process_html_content(self, html_content: str, source: str) -> List[Document]:
        """Process HTML content"""
        return self._process_web_html(html_content, source, include_metadata=False)
    
    def _process_xml_content(self, xml_content: str, source: str) -> List[Document]:
        """Process XML content"""
        try:
            import xml.etree.ElementTree as ET
            
            # Try to parse as XML
            root = ET.fromstring(xml_content)
            
            # Convert to text representation
            def element_to_text(element, path="") -> List[str]:
                parts = []
                current_path = f"{path}/{element.tag}" if path else element.tag
                
                # Add attributes
                if element.attrib:
                    attr_text = ", ".join([f"{k}='{v}'" for k, v in element.attrib.items()])
                    parts.append(f"{current_path}[{attr_text}]: {element.text or ''}")
                else:
                    if element.text and element.text.strip():
                        parts.append(f"{current_path}: {element.text.strip()}")
                
                # Process children
                for child in element:
                    parts.extend(element_to_text(child, current_path))
                
                return parts
            
            content_parts = element_to_text(root)
            text_content = "\n".join(content_parts)
            
            metadata = {
                'source': source,
                'doc_type': 'xml',
                'root_element': root.tag,
                'processing_timestamp': datetime.now().isoformat(),
            }
            
            return self.process_text_content(text_content, source, metadata)
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML content: {str(e)}")
            # Fallback to text processing
            return self.process_text_content(xml_content, source, {'doc_type': 'xml_text'})
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_extensions.keys())
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate if file can be processed"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return False
            
            extension = file_path.suffix.lower()
            return extension in self.supported_extensions
        except Exception:
            return False
    
    def extract_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract basic file information"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        
        return {
            'file_name': file_path.name,
            'file_size': stat.st_size,
            'extension': file_path.suffix.lower(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'is_supported': file_path.suffix.lower() in self.supported_extensions
        }