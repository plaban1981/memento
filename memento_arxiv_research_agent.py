#!/usr/bin/env python3
"""
Memento ArXiv Research Paper Generator
Enhanced version that integrates ArXiv paper reading and writing logic with Memento library

This script combines:
- Memento's hierarchical agent architecture (Meta-Planner + Executor)
- Memory-based learning without weight updates
- ArXiv paper search, reading, and analysis capabilities
- LaTeX paper generation and PDF rendering
"""

import os
import sys
import asyncio
import json
import logging
import io
import subprocess
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# ArXiv and PDF processing imports
import requests
import xml.etree.ElementTree as ET
import PyPDF2
import markdown

# WeasyPrint will be imported inside functions to avoid Windows dependency issues
WEASYPRINT_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

# Add Memento to Python path
MEMENTO_PATH = Path(__file__).parent / "Memento"
sys.path.insert(0, str(MEMENTO_PATH / "client"))
sys.path.insert(0, str(MEMENTO_PATH / "server"))

# Import Memento components
from agent import HierarchicalClient, OpenAIBackend, trim_messages
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ARXIV PAPER PROCESSING LOGIC (from AIwithhassan's implementation)
# ============================================================================

def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    """Search arXiv papers using the exact logic from AIwithhassan's implementation"""
    query = "+".join(topic.lower().split())
    for char in list('()" '):
        if char in query:
            print(f"Invalid character '{char}' in query: {query}")
            raise ValueError(f"Cannot have character: '{char}' in query: {query}")
    
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )
    
    print(f"Making request to arXiv API: {url}")
    resp = requests.get(url)
    
    if not resp.ok:
        print(f"ArXiv API request failed: {resp.status_code} - {resp.text}")
        raise ValueError(f"Bad response from arXiv API: {resp}\n{resp.text}")
    
    data = parse_arxiv_xml(resp.text)
    return data

def parse_arxiv_xml(xml_content: str) -> dict:
    """Parse the XML content from arXiv API response (from AIwithhassan's implementation)"""
    entries = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    root = ET.fromstring(xml_content)
    
    # Loop through each <entry> in Atom namespace
    for entry in root.findall("atom:entry", ns):
        # Extract authors
        authors = [
            author.findtext("atom:name", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        
        # Extract categories (term attribute)
        categories = [
            cat.attrib.get("term")
            for cat in entry.findall("atom:category", ns)
        ]
        
        # Extract PDF link (rel="related" and type="application/pdf")
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        entries.append({
            "title": entry.findtext("atom:title", namespaces=ns),
            "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
            "authors": authors,
            "categories": categories,
            "pdf": pdf_link
        })

    return {"entries": entries}

def read_pdf_from_url(url: str) -> str:
    """Read and extract text from a PDF file given its URL (from AIwithhassan's implementation)"""
    try:
        response = requests.get(url)
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for i, page in enumerate(pdf_reader.pages, 1):
            print(f"Extracting text from page {i}/{num_pages}")
            text += page.extract_text() + "\n"

        print(f"Successfully extracted {len(text)} characters of text from PDF")
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        raise

def convert_markdown_to_pdf(markdown_file_path: str, output_pdf_path: str = None) -> str:
    """
    Convert a markdown file to PDF format
    """
    try:
        # Read the markdown file
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Generate output path if not provided
        if output_pdf_path is None:
            base_path = Path(markdown_file_path).stem
            output_dir = Path(markdown_file_path).parent
            output_pdf_path = output_dir / f"{base_path}.pdf"
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc', 'codehilite']
        )
        
        # Add CSS styling for better PDF appearance
        css_style = """
        <style>
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            margin: 1in;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        h1 { font-size: 24pt; border-bottom: 2px solid #3498db; }
        h2 { font-size: 18pt; border-bottom: 1px solid #bdc3c7; }
        h3 { font-size: 14pt; }
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin-left: 0;
            padding-left: 20px;
            font-style: italic;
            color: #555;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .page-break {
            page-break-before: always;
        }
        </style>
        """
        
        # Combine CSS and HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Research Paper</title>
            {css_style}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Try WeasyPrint first (better quality)
        try:
            from weasyprint import HTML, CSS
            HTML(string=full_html).write_pdf(output_pdf_path)
            logger.info(f"✅ PDF generated using WeasyPrint: {output_pdf_path}")
            return str(output_pdf_path)
        except ImportError:
            logger.info("WeasyPrint not available due to Windows library dependencies")
        except Exception as weasy_error:
            logger.warning(f"WeasyPrint failed: {weasy_error}")
            
            # Fallback to pdfkit (requires wkhtmltopdf)
            if PDFKIT_AVAILABLE:
                try:
                    options = {
                        'page-size': 'A4',
                        'margin-top': '1in',
                        'margin-right': '1in',
                        'margin-bottom': '1in',
                        'margin-left': '1in',
                        'encoding': "UTF-8",
                        'no-outline': None,
                        'enable-local-file-access': None
                    }
                    pdfkit.from_string(full_html, output_pdf_path, options=options)
                    logger.info(f"✅ PDF generated using pdfkit: {output_pdf_path}")
                    return str(output_pdf_path)
                except Exception as pdfkit_error:
                    logger.error(f"pdfkit also failed: {pdfkit_error}")
            else:
                logger.warning("pdfkit not available")
                
                # Final fallback: save as HTML
                html_path = str(output_pdf_path).replace('.pdf', '.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                logger.info(f"⚠️ Saved as HTML instead: {html_path}")
                return html_path
    
    except Exception as e:
        error_msg = f"Error converting markdown to PDF: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def render_latex_pdf(latex_content: str) -> str:
    """Render a LaTeX document to PDF (from AIwithhassan's implementation)"""
    if shutil.which("tectonic") is None:
        raise RuntimeError(
            "tectonic is not installed. Install it first on your system."
        )

    try:
        # Create directory
        output_dir = Path("arxiv_research_output").absolute()
        output_dir.mkdir(exist_ok=True)
        
        # Setup filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tex_filename = f"arxiv_paper_{timestamp}.tex"
        pdf_filename = f"arxiv_paper_{timestamp}.pdf"
        
        # Export as tex & pdf
        tex_file = output_dir / tex_filename
        tex_file.write_text(latex_content)

        result = subprocess.run(
            ["tectonic", tex_filename, "--outdir", str(output_dir)],
            cwd=output_dir,
            capture_output=True,
            text=True,
        )

        final_pdf = output_dir / pdf_filename
        if not final_pdf.exists():
            raise FileNotFoundError("PDF file was not generated")

        print(f"Successfully generated PDF at {final_pdf}")
        return str(final_pdf)

    except Exception as e:
        print(f"Error rendering LaTeX: {str(e)}")
        raise

# ============================================================================
# MEMENTO MEMORY SYSTEM FOR ARXIV RESEARCH
# ============================================================================

@dataclass
class ArXivResearchCase:
    """Represents an ArXiv research case in Memento's memory system"""
    query: str
    domain: str
    arxiv_papers_found: int
    papers_analyzed: List[str]  # Paper titles
    key_insights: List[str]
    research_gaps_identified: List[str]
    novel_contributions: List[str]
    success_metrics: Dict[str, float]
    timestamp: datetime
    final_paper_generated: bool
    latex_quality_score: float

class MementoArXivMemorySystem:
    """Enhanced memory system specifically for ArXiv research paper generation"""
    
    def __init__(self, storage_path: str = "memento_arxiv_memory.json"):
        self.storage_path = storage_path
        self.cases: List[ArXivResearchCase] = []
        self.load_cases()
    
    def store_case(self, case: ArXivResearchCase):
        """Store a new ArXiv research case"""
        self.cases.append(case)
        self.save_cases()
        logger.info(f"Stored ArXiv research case: {case.query[:50]}...")
    
    def retrieve_similar_cases(self, query: str, domain: str = "", k: int = 4) -> List[ArXivResearchCase]:
        """Retrieve K most similar ArXiv research cases (Memento's optimal K=4)"""
        query_words = set(query.lower().split())
        domain_words = set(domain.lower().split()) if domain else set()
        
        scored_cases = []
        for case in self.cases:
            case_words = set(case.query.lower().split())
            case_domain_words = set(case.domain.lower().split())
            
            # Calculate similarity based on query, domain, and research patterns
            query_similarity = len(query_words.intersection(case_words)) / len(query_words.union(case_words))
            domain_similarity = len(domain_words.intersection(case_domain_words)) / len(domain_words.union(case_domain_words)) if domain_words else 0
            
            # Bonus for successful cases with high paper quality
            success_bonus = case.success_metrics.get('completion_score', 0) * 0.1
            
            # Combined similarity (60% query, 30% domain, 10% success)
            similarity = 0.6 * query_similarity + 0.3 * domain_similarity + success_bonus
            scored_cases.append((similarity, case))
        
        # Return top K cases
        scored_cases.sort(key=lambda x: x[0], reverse=True)
        return [case for _, case in scored_cases[:k]]
    
    def save_cases(self):
        """Save cases to disk"""
        data = []
        for case in self.cases:
            case_dict = asdict(case)
            case_dict['timestamp'] = case_dict['timestamp'].isoformat()
            data.append(case_dict)
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_cases(self):
        """Load cases from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.cases = []
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    self.cases.append(ArXivResearchCase(**item))
                
                logger.info(f"Loaded {len(self.cases)} ArXiv research cases from memory")
            except Exception as e:
                logger.error(f"Error loading ArXiv cases: {e}")
                self.cases = []

# ============================================================================
# ARXIV-SPECIFIC RESEARCH PROMPTS
# ============================================================================

ARXIV_META_PROMPT = """
You are the META-PLANNER for an ArXiv-based research paper generation system.
Your role is to break down research queries into executable tasks for academic paper generation using ArXiv papers.

When given a research topic, create a plan that includes:
1. ArXiv paper search for the specific topic
2. Selection and analysis of the most relevant papers
3. PDF content extraction and detailed reading
4. Research gap identification from existing literature
5. Novel contribution formulation
6. Comprehensive paper writing with proper citations
7. LaTeX formatting and PDF generation

Reply ONLY in JSON with the schema:
{ "plan": [ {"id": INT, "description": STRING} ... ] }

Focus on:
- Finding recent ArXiv papers (last 2-3 years preferred)
- Analyzing paper methodologies and results
- Identifying clear research gaps
- Proposing novel solutions or improvements
- Creating publication-ready LaTeX documents

If the research is complete, output: FINAL ANSWER: <complete_paper_content>

⚠️ Reply with *pure JSON only*.
"""

ARXIV_EXEC_PROMPT = """
You are the EXECUTOR for ArXiv-based research paper generation.
You receive research tasks and execute them using ArXiv search and analysis capabilities.

Your primary workflow:
1. Search ArXiv for papers using specific keywords
2. Analyze paper abstracts and select most relevant ones
3. Read full PDF content of selected papers
4. Extract key methodologies, results, and limitations
5. Identify research gaps and opportunities
6. Generate novel research contributions
7. Write comprehensive academic papers with proper citations
8. Format papers in LaTeX for publication

Always:
- Prioritize recent ArXiv papers (2022-2024)
- Read full paper content, not just abstracts
- Cite papers properly with ArXiv IDs
- Identify specific methodological gaps
- Propose concrete, implementable solutions
- Generate publication-ready LaTeX code

Use available tools efficiently and provide detailed, well-structured academic results.
"""

# ============================================================================
# MEMENTO ARXIV RESEARCH AGENT
# ============================================================================

class MementoArXivResearchAgent:
    """ArXiv research paper generator using Memento's hierarchical architecture"""
    
    def __init__(self, 
                 meta_model: str = "o3",
                 exec_model: str = "o3-2025-04-16"):
        
        # Initialize ArXiv-specific memory system
        self.memory = MementoArXivMemorySystem()
        
        # Initialize Memento client
        self.client = HierarchicalClient(meta_model, exec_model)
        
        # Override system prompts for ArXiv research focus
        self.arxiv_meta_prompt = ARXIV_META_PROMPT
        self.arxiv_exec_prompt = ARXIV_EXEC_PROMPT
        
        self.connected = False
    
    async def initialize(self):
        """Initialize the agent - ArXiv tools are built-in, no MCP servers needed"""
        if not self.connected:
            # For ArXiv research, we use built-in tools rather than MCP servers
            logger.info("✅ Memento ArXiv Research Agent initialized (using built-in ArXiv tools)")
            self.connected = True
    
    def consume_research_paper(self, markdown_file_path: str) -> Dict[str, Any]:
        """
        Consume a research paper from memento_library_research_agent output
        and extract key information for ArXiv research enhancement
        """
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from the markdown file
            lines = content.split('\n')
            metadata = {}
            
            for line in lines[:20]:  # Check first 20 lines for metadata
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                elif line.startswith('**Domain:**'):
                    metadata['domain'] = line.split('**Domain:**')[1].strip()
                elif line.startswith('**Generated:**'):
                    metadata['generated'] = line.split('**Generated:**')[1].strip()
                elif line.startswith('**Tools Used:**'):
                    metadata['tools_used'] = line.split('**Tools Used:**')[1].strip()
            
            # Extract the main content (after the metadata section)
            content_start = content.find('---\n')
            if content_start != -1:
                main_content = content[content_start + 4:].strip()
            else:
                main_content = content
            
            # Extract research topic from title
            research_topic = metadata.get('title', '').replace('Research Paper: ', '').strip()
            
            logger.info(f"📄 Consumed research paper: {research_topic}")
            logger.info(f"🏷️  Domain: {metadata.get('domain', 'Unknown')}")
            
            return {
                "success": True,
                "metadata": metadata,
                "content": main_content,
                "research_topic": research_topic,
                "domain": metadata.get('domain', 'computer science'),
                "file_path": markdown_file_path
            }
            
        except Exception as e:
            logger.error(f"❌ Error consuming research paper: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": markdown_file_path
            }
    
    async def enhance_research_with_arxiv(self, 
                                        consumed_paper: Dict[str, Any], 
                                        max_arxiv_papers: int = 10) -> Dict[str, Any]:
        """
        Enhance a consumed research paper with ArXiv literature review and analysis
        """
        if not consumed_paper.get("success"):
            return consumed_paper
        
        research_topic = consumed_paper["research_topic"]
        domain = consumed_paper["domain"]
        existing_content = consumed_paper["content"]
        
        logger.info(f"🔬 Enhancing research paper with ArXiv literature for: {research_topic}")
        
        # Search ArXiv for related papers
        arxiv_analysis = await self.search_and_analyze_arxiv_papers(research_topic, max_arxiv_papers)
        
        if not arxiv_analysis.get("success"):
            logger.warning("⚠️ ArXiv search failed, proceeding with original content")
            arxiv_papers = []
        else:
            arxiv_papers = arxiv_analysis["papers"]
        
        # Create enhanced research query combining original content with ArXiv findings
        enhancement_query = f"""
        RESEARCH ENHANCEMENT TASK:
        
        Original Research Topic: {research_topic}
        Domain: {domain}
        
        EXISTING RESEARCH CONTENT:
        {existing_content[:2000]}...
        
        ARXIV LITERATURE FINDINGS:
        """
        
        if arxiv_papers:
            enhancement_query += f"\nFound {len(arxiv_papers)} relevant ArXiv papers:\n"
            for i, paper in enumerate(arxiv_papers[:5], 1):
                enhancement_query += f"\n{i}. {paper['title']}\n"
                enhancement_query += f"   Authors: {', '.join(paper['authors'])}\n"
                enhancement_query += f"   ArXiv ID: {paper['arxiv_id']}\n"
                enhancement_query += f"   Summary: {paper['summary'][:200]}...\n"
                if paper.get('methodology'):
                    enhancement_query += f"   Methodology: {paper['methodology']}\n"
                if paper.get('contributions'):
                    enhancement_query += f"   Contributions: {', '.join(paper['contributions'][:2])}\n"
        else:
            enhancement_query += "\nNo ArXiv papers found for this topic.\n"
        
        enhancement_query += f"""
        
        ENHANCEMENT REQUIREMENTS:
        1. Integrate ArXiv findings into the existing research
        2. Add a comprehensive Literature Review section
        3. Identify research gaps from ArXiv papers
        4. Enhance methodology based on recent ArXiv work
        5. Add proper ArXiv citations throughout
        6. Strengthen the contribution section
        7. Generate publication-ready LaTeX format
        8. Include references section with ArXiv papers
        
        Create an enhanced, comprehensive research paper that combines the original research with ArXiv literature insights.
        """
        
        return {
            "success": True,
            "enhancement_query": enhancement_query,
            "original_content": existing_content,
            "arxiv_papers": arxiv_papers,
            "research_topic": research_topic,
            "domain": domain,
            "metadata": consumed_paper["metadata"]
        }

    async def search_and_analyze_arxiv_papers(self, topic: str, max_results: int = 10) -> Dict[str, Any]:
        """Search ArXiv papers and analyze them"""
        logger.info(f"🔍 Searching ArXiv for: {topic}")
        
        try:
            # Search ArXiv papers
            papers_data = search_arxiv_papers(topic, max_results)
            papers = papers_data.get("entries", [])
            
            if not papers:
                return {"success": False, "error": "No papers found", "papers": []}
            
            logger.info(f"📄 Found {len(papers)} papers")
            
            # Analyze and select most relevant papers
            analyzed_papers = []
            for i, paper in enumerate(papers[:5], 1):  # Analyze top 5 papers
                logger.info(f"📖 Analyzing paper {i}/5: {paper['title'][:60]}...")
                
                paper_analysis = {
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "summary": paper["summary"],
                    "categories": paper["categories"],
                    "pdf_url": paper["pdf"],
                    "content": None,
                    "key_insights": [],
                    "methodology": "",
                    "limitations": "",
                    "relevance_score": 0.0
                }
                
                # Read PDF content if available
                if paper["pdf"]:
                    try:
                        logger.info(f"📑 Reading PDF content...")
                        pdf_content = read_pdf_from_url(paper["pdf"])
                        paper_analysis["content"] = pdf_content[:5000]  # First 5000 chars
                        
                        # Extract key insights using LLM
                        insights_prompt = f"""
                        Analyze this ArXiv paper and extract:
                        1. Key methodology used
                        2. Main contributions
                        3. Limitations mentioned
                        4. Relevance to topic "{topic}" (score 0-1)
                        
                        Paper Title: {paper['title']}
                        Content: {pdf_content[:3000]}
                        
                        Respond in JSON format:
                        {{
                            "methodology": "brief description",
                            "contributions": ["contrib1", "contrib2"],
                            "limitations": ["limit1", "limit2"],
                            "relevance_score": 0.85
                        }}
                        """
                        
                        # Use Memento's LLM backend for analysis
                        analysis_result = await self.client.meta_llm.chat([
                            {"role": "user", "content": insights_prompt}
                        ])
                        
                        try:
                            analysis_json = json.loads(analysis_result["content"])
                            paper_analysis.update(analysis_json)
                        except:
                            logger.warning(f"Could not parse analysis for paper {i}")
                        
                    except Exception as e:
                        logger.warning(f"Could not read PDF for paper {i}: {e}")
                
                analyzed_papers.append(paper_analysis)
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(1)
            
            return {
                "success": True,
                "papers": analyzed_papers,
                "total_found": len(papers),
                "analyzed": len(analyzed_papers)
            }
            
        except Exception as e:
            logger.error(f"Error in ArXiv search and analysis: {e}")
            return {"success": False, "error": str(e), "papers": []}
    
    async def generate_arxiv_research_paper(self, 
                                          research_query: str,
                                          domain: str = "computer science",
                                          max_arxiv_papers: int = 10) -> Dict[str, Any]:
        """Generate a research paper based on ArXiv papers using Memento's hierarchical approach"""
        
        if not self.connected:
            await self.initialize()
        
        logger.info(f"🔬 Starting ArXiv research paper generation: {research_query}")
        
        # Retrieve similar cases from memory
        similar_cases = self.memory.retrieve_similar_cases(research_query, domain)
        
        # Search and analyze ArXiv papers
        arxiv_analysis = await self.search_and_analyze_arxiv_papers(research_query, max_arxiv_papers)
        
        if not arxiv_analysis["success"]:
            return {
                "success": False,
                "error": f"ArXiv search failed: {arxiv_analysis['error']}"
            }
        
        analyzed_papers = arxiv_analysis["papers"]
        
        # Prepare enhanced query with memory context and ArXiv papers
        memory_context = ""
        if similar_cases:
            memory_context = "\n\nPrevious ArXiv research experience context:\n"
            for i, case in enumerate(similar_cases[:2], 1):
                memory_context += f"{i}. Previous research on '{case.query}' in {case.domain}\n"
                memory_context += f"   - ArXiv papers analyzed: {case.arxiv_papers_found}\n"
                memory_context += f"   - Key insights: {', '.join(case.key_insights[:2])}\n"
                memory_context += f"   - Research gaps found: {', '.join(case.research_gaps_identified[:2])}\n"
        
        # Prepare ArXiv papers context
        arxiv_context = f"\n\nArXiv Papers Analysis (Found {arxiv_analysis['total_found']}, Analyzed {arxiv_analysis['analyzed']}):\n"
        for i, paper in enumerate(analyzed_papers, 1):
            arxiv_context += f"\n{i}. {paper['title']}\n"
            arxiv_context += f"   Authors: {', '.join(paper['authors'][:3])}\n"
            arxiv_context += f"   Categories: {', '.join(paper['categories'])}\n"
            arxiv_context += f"   Summary: {paper['summary'][:200]}...\n"
            if paper.get('methodology'):
                arxiv_context += f"   Methodology: {paper['methodology']}\n"
            if paper.get('limitations'):
                arxiv_context += f"   Limitations: {', '.join(paper['limitations'][:2])}\n"
            arxiv_context += f"   PDF: {paper['pdf_url']}\n"
        
        enhanced_query = f"""
Research Topic: {research_query}
Domain: {domain}

Task: Generate a comprehensive academic research paper based on recent ArXiv literature

Requirements:
1. Analyze the provided ArXiv papers thoroughly
2. Identify clear research gaps from the existing literature
3. Propose novel contributions or improvements
4. Write a publication-ready research paper with:
   - Abstract with clear contributions
   - Introduction with motivation
   - Related Work section citing ArXiv papers
   - Methodology section with technical details
   - Results/Analysis section
   - Discussion of limitations and future work
   - Conclusion
   - References with proper ArXiv citations
5. Format the paper in LaTeX for publication
6. Ensure mathematical equations are properly formatted
7. Include figures/tables if relevant

{memory_context}

{arxiv_context}

Please create a detailed, publication-ready research paper that builds upon these ArXiv papers and makes novel contributions to the field.
        """.strip()
        
        try:
            # Use Memento's hierarchical processing with custom prompts
            original_meta_prompt = self.client.shared_history
            
            # Process with ArXiv-specific prompts
            result = await self.client.process_query(
                query=enhanced_query,
                file="",
                task_id=f"arxiv_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Extract insights and metrics
            papers_analyzed = [p["title"] for p in analyzed_papers]
            
            # Identify research gaps and contributions from the result
            research_gaps = []
            novel_contributions = []
            
            if "gap" in result.lower() or "limitation" in result.lower():
                research_gaps.append("Research gaps identified from ArXiv analysis")
            if "novel" in result.lower() or "contribution" in result.lower():
                novel_contributions.append("Novel contributions proposed")
            if "improve" in result.lower() or "enhance" in result.lower():
                novel_contributions.append("Improvements to existing methods")
            
            # Calculate success metrics
            success_metrics = {
                "completion_score": 1.0 if "FINAL ANSWER:" in result else 0.8,
                "arxiv_papers_utilized": len(analyzed_papers) / max_arxiv_papers,
                "content_quality": min(1.0, len(result) / 8000.0),
                "latex_detected": 1.0 if ("\\documentclass" in result or "\\begin{document}" in result) else 0.0,
                "citations_included": 1.0 if "arxiv" in result.lower() else 0.5
            }
            
            latex_quality_score = success_metrics["latex_detected"]
            
            # Store the ArXiv research case in memory
            arxiv_case = ArXivResearchCase(
                query=research_query,
                domain=domain,
                arxiv_papers_found=arxiv_analysis["total_found"],
                papers_analyzed=papers_analyzed,
                key_insights=["ArXiv literature analysis", "Research gap identification"],
                research_gaps_identified=research_gaps,
                novel_contributions=novel_contributions,
                success_metrics=success_metrics,
                timestamp=datetime.now(),
                final_paper_generated="FINAL ANSWER:" in result,
                latex_quality_score=latex_quality_score
            )
            
            self.memory.store_case(arxiv_case)
            
            # Save results to files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("arxiv_research_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Save full result in markdown format
            result_file = output_dir / f"arxiv_research_paper_{timestamp}.md"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"# ArXiv Research Paper: {research_query}\n\n")
                f.write(f"**Domain:** {domain}\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**ArXiv Papers Analyzed:** {len(analyzed_papers)}\n\n")
                f.write("## ArXiv Papers Used:\n")
                for paper in analyzed_papers:
                    f.write(f"- {paper['title']} ({paper['pdf_url']})\n")
                f.write("\n---\n\n")
                f.write(result)
            
            # Also save as PDF using the converter function
            html_pdf_file = None
            try:
                html_pdf_file = convert_markdown_to_html_pdf(str(result_file))
                if html_pdf_file:
                    logger.info(f"📄 PDF-ready HTML created: {html_pdf_file}")
            except Exception as e:
                logger.warning(f"Could not create PDF version: {e}")
            
            # Save LaTeX if present and render PDF
            latex_file = None
            pdf_file = None
            if "\\documentclass" in result or "\\begin{document}" in result:
                latex_file = output_dir / f"arxiv_research_paper_{timestamp}.tex"
                # Extract LaTeX content
                latex_start = result.find("\\documentclass")
                if latex_start != -1:
                    latex_content = result[latex_start:]
                    # Find end of LaTeX document
                    latex_end = latex_content.find("\\end{document}")
                    if latex_end != -1:
                        latex_content = latex_content[:latex_end + len("\\end{document}")]
                    
                    with open(latex_file, 'w', encoding='utf-8') as f:
                        f.write(latex_content)
                    
                    # Try to render PDF
                    try:
                        pdf_file = render_latex_pdf(latex_content)
                    except Exception as e:
                        logger.warning(f"Could not render PDF: {e}")
            
            logger.info(f"✅ ArXiv research paper generated successfully!")
            logger.info(f"📄 Markdown saved to: {result_file}")
            if html_pdf_file:
                logger.info(f"📄 PDF-ready HTML saved to: {html_pdf_file}")
            if latex_file:
                logger.info(f"📊 LaTeX saved to: {latex_file}")
            if pdf_file:
                logger.info(f"🎯 PDF generated at: {pdf_file}")
            
            return {
                "success": True,
                "result": result,
                "research_query": research_query,
                "domain": domain,
                "arxiv_papers_analyzed": len(analyzed_papers),
                "papers_details": analyzed_papers,
                "success_metrics": success_metrics,
                "research_gaps": research_gaps,
                "novel_contributions": novel_contributions,
                "result_file": str(result_file),
                "html_pdf_file": html_pdf_file,
                "latex_file": str(latex_file) if latex_file else None,
                "pdf_file": pdf_file,
                "memory_cases_used": len(similar_cases)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating ArXiv research paper: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "research_query": research_query
            }
    
    async def interactive_arxiv_research_session(self):
        """Interactive ArXiv research session using Memento"""
        
        if not self.connected:
            await self.initialize()
        
        print("🔬 Memento ArXiv Research Paper Generator")
        print("=" * 55)
        print("Specialized for ArXiv paper analysis and research paper generation")
        print("Type 'quit' to exit, 'memory' to view stored cases, 'papers' to see recent ArXiv papers")
        
        while True:
            try:
                query = input("\n📝 Research topic: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.lower() == 'memory':
                    print(f"\n💾 ArXiv Memory contains {len(self.memory.cases)} research cases:")
                    for i, case in enumerate(self.memory.cases[-5:], 1):  # Show last 5
                        print(f"{i}. {case.query} ({case.domain})")
                        print(f"   - ArXiv papers: {case.arxiv_papers_found}, Success: {case.success_metrics.get('completion_score', 0):.2f}")
                        print(f"   - Gaps found: {len(case.research_gaps_identified)}, Contributions: {len(case.novel_contributions)}")
                    continue
                
                if query.lower() == 'papers':
                    topic = input("🔍 Enter topic to search ArXiv: ").strip()
                    if topic:
                        papers_result = await self.search_and_analyze_arxiv_papers(topic, 5)
                        if papers_result["success"]:
                            print(f"\n📄 Found {papers_result['total_found']} papers, analyzed {papers_result['analyzed']}:")
                            for i, paper in enumerate(papers_result["papers"], 1):
                                print(f"{i}. {paper['title'][:80]}...")
                                print(f"   Authors: {', '.join(paper['authors'][:2])}")
                                print(f"   Relevance: {paper.get('relevance_score', 'N/A')}")
                        else:
                            print(f"❌ Search failed: {papers_result['error']}")
                    continue
                
                if not query:
                    print("❌ Please enter a research topic")
                    continue
                
                domain = input("🏷️  Research domain (default: computer science): ").strip()
                if not domain:
                    domain = "computer science"
                
                max_papers = input("📊 Max ArXiv papers to analyze (default: 10): ").strip()
                try:
                    max_papers = int(max_papers) if max_papers else 10
                except:
                    max_papers = 10
                
                print(f"\n🚀 Generating ArXiv-based research paper...")
                print("This will search ArXiv, analyze papers, and generate a comprehensive research paper...")
                print("This may take several minutes...")
                
                result = await self.generate_arxiv_research_paper(query, domain, max_papers)
                
                if result["success"]:
                    print(f"\n✅ ArXiv research paper generated!")
                    print(f"📊 ArXiv papers analyzed: {result['arxiv_papers_analyzed']}")
                    print(f"🎯 Success score: {result['success_metrics']['completion_score']:.2f}")
                    print(f"🔍 Research gaps found: {len(result['research_gaps'])}")
                    print(f"💡 Novel contributions: {len(result['novel_contributions'])}")
                    print(f"📄 Saved to: {result['result_file']}")
                    if result.get("pdf_file"):
                        print(f"📊 PDF generated: {result['pdf_file']}")
                    
                    # Show ArXiv papers used
                    show_papers = input("\n📚 Show ArXiv papers analyzed? (y/n): ").strip().lower()
                    if show_papers == 'y':
                        print(f"\n📄 ArXiv Papers Analyzed:")
                        for i, paper in enumerate(result["papers_details"], 1):
                            print(f"{i}. {paper['title']}")
                            print(f"   Authors: {', '.join(paper['authors'][:3])}")
                            print(f"   URL: {paper['pdf_url']}")
                    
                    # Show preview
                    preview = result["result"][:800] + "..." if len(result["result"]) > 800 else result["result"]
                    show_preview = input("\n👀 Show paper preview? (y/n): ").strip().lower()
                    if show_preview == 'y':
                        print(f"\n📖 Research Paper Preview:")
                        print("-" * 70)
                        print(preview)
                        print("-" * 70)
                else:
                    print(f"❌ Failed: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.connected:
            await self.client.cleanup()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def convert_research_paper_to_pdf():
    """Convert a specific research paper markdown file to PDF"""
    
    # Default file path
    markdown_file = r"C:\Users\nayak\Documents\Agent_Fly\research_outputs\research_conversation_20250831_103320.md"
    
    print("📄 Research Paper to PDF Converter")
    print("=" * 50)
    
    # Ask for file path
    file_path = input(f"📁 Enter markdown file path\n(default: {markdown_file}): ").strip()
    
    if not file_path:
        file_path = markdown_file
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Generate output PDF path
    output_pdf = file_path.replace('.md', '.pdf')
    
    print(f"\n🔄 Converting to PDF...")
    print(f"📄 Input:  {file_path}")
    print(f"📄 Output: {output_pdf}")
    
    try:
        # Convert to PDF
        result_path = convert_markdown_to_pdf(file_path, output_pdf)
        
        if result_path.endswith('.pdf'):
            print(f"\n✅ Successfully converted to PDF!")
            print(f"📄 PDF saved at: {result_path}")
            
            # Get file size
            file_size = os.path.getsize(result_path) / 1024  # KB
            print(f"📊 File size: {file_size:.1f} KB")
            
        else:
            print(f"\n⚠️ Converted to HTML format instead of PDF")
            print(f"📄 HTML saved at: {result_path}")
            print("💡 Install WeasyPrint or wkhtmltopdf for PDF generation")
    
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Install required packages: pip install markdown weasyprint")
        print("   - Or install wkhtmltopdf for pdfkit support")

async def enhance_existing_research_paper():
    """Enhance an existing research paper with ArXiv literature"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please set GROQ_API_KEY in your .env file")
        return
    
    agent = MementoArXivResearchAgent()
    
    print("📄 Research Paper Enhancement with ArXiv Literature")
    print("=" * 60)
    
    # Ask for the markdown file path
    default_path = "memento_research_outputs/research_paper_20250831_105044.md"
    file_path = input(f"📁 Enter path to research paper markdown file\n(default: {default_path}): ").strip()
    
    if not file_path:
        file_path = default_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n🔍 Consuming research paper: {file_path}")
    
    # Consume the existing research paper
    consumed_paper = agent.consume_research_paper(file_path)
    
    if not consumed_paper["success"]:
        print(f"❌ Failed to consume research paper: {consumed_paper['error']}")
        return
    
    print(f"✅ Successfully consumed: {consumed_paper['research_topic']}")
    print(f"🏷️  Domain: {consumed_paper['domain']}")
    
    # Ask for number of ArXiv papers to search
    max_papers = input("\n📊 Number of ArXiv papers to search (default: 10): ").strip()
    try:
        max_papers = int(max_papers) if max_papers else 10
    except ValueError:
        max_papers = 10
    
    print(f"\n🚀 Enhancing with ArXiv literature (searching {max_papers} papers)...")
    
    try:
        # Enhance with ArXiv literature
        enhancement_data = await agent.enhance_research_with_arxiv(consumed_paper, max_papers)
        
        if not enhancement_data["success"]:
            print(f"❌ Enhancement failed: {enhancement_data.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Found {len(enhancement_data['arxiv_papers'])} relevant ArXiv papers")
        
        # Generate enhanced research paper using the enhancement query
        enhanced_result = await agent.generate_arxiv_research_paper(
            enhancement_data["research_topic"],
            enhancement_data["domain"],
            max_arxiv_papers=0  # Skip ArXiv search since we already have the data
        )
        
        if enhanced_result["success"]:
            print(f"\n🎉 Enhanced research paper generated!")
            print(f"📄 Original file: {file_path}")
            print(f"📄 Enhanced file: {enhanced_result['result_file']}")
            if enhanced_result.get("latex_file"):
                print(f"📊 LaTeX file: {enhanced_result['latex_file']}")
            
            # Show comparison
            print(f"\n📊 Enhancement Summary:")
            print(f"   - Original content length: {len(consumed_paper['content'])} chars")
            print(f"   - Enhanced content length: {len(enhanced_result['result'])} chars")
            print(f"   - ArXiv papers integrated: {len(enhancement_data['arxiv_papers'])}")
            
        else:
            print(f"❌ Failed to generate enhanced paper: {enhanced_result['error']}")
    
    except Exception as e:
        print(f"❌ Error during enhancement: {e}")
    
    finally:
        await agent.cleanup()

async def main():
    """Main function with ArXiv research options"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please set GROQ_API_KEY in your .env file")
        return
    
    print("🚀 Using GROQ API with openai/gpt-oss-20b model")
    print("🔬 Memento ArXiv Research Paper Generator")
    print("Using Memento's Hierarchical Architecture + ArXiv Integration")
    print("=" * 65)
    print("\nOptions:")
    print("1. Interactive ArXiv research session")
    print("2. Single ArXiv research query")
    print("3. Search and analyze ArXiv papers only")
    print("4. Convert markdown research paper to PDF")
    print("5. Enhance existing research paper with ArXiv")
    print("6. Exit")
    
    while True:
        choice = input("\n🔢 Choose an option (1-6): ").strip()
        
        if choice == '1':
            agent = MementoArXivResearchAgent()
            try:
                await agent.interactive_arxiv_research_session()
            finally:
                await agent.cleanup()
            break
            
        elif choice == '2':
            query = input("\n📝 Enter research query: ").strip()
            if not query:
                print("❌ Please enter a valid query")
                continue
                
            domain = input("🏷️  Research domain (default: computer science): ").strip()
            if not domain:
                domain = "computer science"
            
            max_papers = input("📊 Max ArXiv papers to analyze (default: 10): ").strip()
            try:
                max_papers = int(max_papers) if max_papers else 10
            except:
                max_papers = 10
            
            agent = MementoArXivResearchAgent()
            try:
                result = await agent.generate_arxiv_research_paper(query, domain, max_papers)
                
                if result["success"]:
                    print(f"\n✅ ArXiv research paper generated!")
                    print(f"📄 Saved to: {result['result_file']}")
                    print(f"📊 ArXiv papers analyzed: {result['arxiv_papers_analyzed']}")
                    if result.get("pdf_file"):
                        print(f"🎯 PDF generated: {result['pdf_file']}")
                else:
                    print(f"❌ Failed: {result['error']}")
            finally:
                await agent.cleanup()
            break
            
        elif choice == '3':
            topic = input("\n🔍 Enter topic to search ArXiv: ").strip()
            if not topic:
                print("❌ Please enter a valid topic")
                continue
            
            max_results = input("📊 Max results (default: 10): ").strip()
            try:
                max_results = int(max_results) if max_results else 10
            except:
                max_results = 10
            
            agent = MementoArXivResearchAgent()
            try:
                await agent.initialize()
                result = await agent.search_and_analyze_arxiv_papers(topic, max_results)
                
                if result["success"]:
                    print(f"\n✅ Found {result['total_found']} papers, analyzed {result['analyzed']}")
                    for i, paper in enumerate(result["papers"], 1):
                        print(f"\n{i}. {paper['title']}")
                        print(f"   Authors: {', '.join(paper['authors'][:3])}")
                        print(f"   Categories: {', '.join(paper['categories'])}")
                        print(f"   Relevance: {paper.get('relevance_score', 'N/A')}")
                        print(f"   PDF: {paper['pdf_url']}")
                        if paper.get('methodology'):
                            print(f"   Methodology: {paper['methodology']}")
                else:
                    print(f"❌ Search failed: {result['error']}")
            finally:
                await agent.cleanup()
            break
            
        elif choice == '4':
            await convert_research_paper_to_pdf()
            break
            
        elif choice == '5':
            await enhance_existing_research_paper()
            break
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    asyncio.run(main()) 