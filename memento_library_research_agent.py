#!/usr/bin/env python3
"""
Memento Library-Based Research Paper Generator
Uses the actual Memento library from the cloned repository

This script demonstrates how to use Memento as a library to create
a specialized research paper generation agent that combines:
- Memento's hierarchical agent architecture (Meta-Planner + Executor)
- Memory-based learning without weight updates
- MCP tool integration for research capabilities
- Academic paper generation with LaTeX support
"""

import os
import sys
import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
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

# Import LaTeX PDF rendering function (avoiding weasyprint dependency issues)
def render_latex_pdf_local(latex_content: str) -> str:
    """Local implementation of LaTeX to PDF rendering using tectonic"""
    import subprocess
    import shutil
    
    logger.info("🚀 Starting LaTeX to PDF conversion...")
    
    if shutil.which("tectonic") is None:
        raise RuntimeError("tectonic is not installed. Install it first on your system.")

    try:
        # Create directory (use same as main function)
        output_dir = Path("memento_research_outputs").absolute()
        output_dir.mkdir(exist_ok=True)
        
        # Setup filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tex_filename = f"research_paper_{timestamp}.tex"
        pdf_filename = f"research_paper_{timestamp}.pdf"
        
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
# Check if tectonic is available
try:
    import shutil
    import os
    
    # Add Chocolatey to PATH if not present (Windows)
    choco_bin = r"C:\ProgramData\chocolatey\bin"
    if os.path.exists(choco_bin) and choco_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] += f";{choco_bin}"
        logger.info(f"🔧 Added Chocolatey to PATH: {choco_bin}")
    
    tectonic_path = shutil.which("tectonic")
    LATEX_PDF_AVAILABLE = tectonic_path is not None
    render_latex_pdf = render_latex_pdf_local
    if LATEX_PDF_AVAILABLE:
        logger.info(f"✅ Tectonic found at: {tectonic_path} - LaTeX PDF rendering enabled")
    else:
        logger.info("💡 Tectonic not found - LaTeX PDF rendering disabled (install tectonic for PDF generation)")
except Exception as e:
    LATEX_PDF_AVAILABLE = False
    logger.warning(f"LaTeX PDF rendering not available: {e}")

# ============================================================================
# MEMENTO MEMORY SYSTEM (Enhanced Case Bank)
# ============================================================================

@dataclass
class ResearchCase:
    """Represents a research case in Memento's memory system"""
    query: str
    domain: str
    approach: str
    tools_used: List[str]
    success_metrics: Dict[str, float]
    insights: List[str]
    timestamp: datetime
    task_complexity: int
    final_result: str

class MementoMemorySystem:
    """Enhanced memory system for research paper generation"""
    
    def __init__(self, storage_path: str = "memento_research_memory.json"):
        self.storage_path = storage_path
        self.cases: List[ResearchCase] = []
        self.load_cases()
    
    def store_case(self, case: ResearchCase):
        """Store a new research case"""
        self.cases.append(case)
        self.save_cases()
        logger.info(f"Stored research case: {case.query[:50]}...")
    
    def retrieve_similar_cases(self, query: str, domain: str = "", k: int = 4) -> List[ResearchCase]:
        """Retrieve K most similar research cases (Memento's optimal K=4)"""
        query_words = set(query.lower().split())
        domain_words = set(domain.lower().split()) if domain else set()
        
        scored_cases = []
        for case in self.cases:
            case_words = set(case.query.lower().split())
            case_domain_words = set(case.domain.lower().split())
            
            # Calculate similarity based on query and domain overlap
            query_similarity = len(query_words.intersection(case_words)) / len(query_words.union(case_words))
            domain_similarity = len(domain_words.intersection(case_domain_words)) / len(domain_words.union(case_domain_words)) if domain_words else 0
            
            # Combined similarity score
            similarity = 0.7 * query_similarity + 0.3 * domain_similarity
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
                    self.cases.append(ResearchCase(**item))
                
                logger.info(f"Loaded {len(self.cases)} research cases from memory")
            except Exception as e:
                logger.error(f"Error loading cases: {e}")
                self.cases = []

# ============================================================================
# RESEARCH-SPECIFIC PROMPTS
# ============================================================================

RESEARCH_META_PROMPT = """
You are the META-PLANNER for an academic research paper generation system.
Your role is to break down research queries into executable tasks for paper generation.

When given a research topic, create a plan that includes:
1. Literature search and paper collection
2. Document analysis and content extraction  
3. Research gap identification
4. Paper structure planning
5. Content generation with citations
6. LaTeX formatting and PDF generation

Reply ONLY in JSON with the schema:
{ "plan": [ {"id": INT, "description": STRING} ... ] }

Focus on academic rigor, proper citations, and comprehensive coverage.
If the research is complete, output: FINAL ANSWER: <complete_paper_content>

⚠️ Reply with *pure JSON only*.
"""

RESEARCH_EXEC_PROMPT = """
You are the EXECUTOR for academic research paper generation.
You receive research tasks and execute them using available tools.

Available research capabilities:
- search: Find academic papers and web resources
- document_processing: Extract text from PDFs and documents
- code_agent: Generate code, analysis, and LaTeX formatting
- excel_tool: Analyze data and create tables
- image_tool: Process figures and diagrams
- math_tool: Handle mathematical equations and proofs

Always prioritize academic sources, proper citations, and scholarly writing.
Use tools efficiently and provide detailed, well-structured results.
"""

# ============================================================================
# MEMENTO RESEARCH AGENT
# ============================================================================

class MementoResearchAgent:
    """Research paper generator using Memento's hierarchical architecture"""
    
    def __init__(self, 
                 meta_model: str = "o3-mini",
                 exec_model: str = "o3-mini",
                 server_scripts: Optional[List[str]] = None):
        
        # Initialize memory system
        self.memory = MementoMemorySystem()
        
        # Initialize Memento client with research-specific prompts
        self.client = HierarchicalClient(meta_model, exec_model)
        
        # Override system prompts for research focus
        self.client.meta_llm.research_prompt = RESEARCH_META_PROMPT
        self.client.exec_llm.research_prompt = RESEARCH_EXEC_PROMPT
        
        # Default server scripts for research (simplified for testing)
        if server_scripts is None:
            server_scripts = [
                str(MEMENTO_PATH / "server" / "search_tool.py"),
                # Temporarily disabled other tools for debugging
                # str(MEMENTO_PATH / "server" / "documents_tool.py"),
                # str(MEMENTO_PATH / "server" / "code_agent.py"),
                # str(MEMENTO_PATH / "server" / "excel_tool.py"),
                # str(MEMENTO_PATH / "server" / "image_tool.py"),
                # str(MEMENTO_PATH / "server" / "math_tool.py"),
            ]
        
        self.server_scripts = server_scripts
        self.connected = False
    
    async def initialize(self):
        """Initialize the agent and connect to MCP servers"""
        if not self.connected:
            await self.client.connect_to_servers(self.server_scripts)
            self.connected = True
            logger.info("✅ Memento Research Agent initialized successfully")
    
    async def generate_research_paper(self, 
                                    research_query: str,
                                    domain: str = "computer science",
                                    file_path: str = "") -> Dict[str, Any]:
        """Generate a research paper using Memento's hierarchical approach"""
        
        if not self.connected:
            await self.initialize()
        
        logger.info(f"🔬 Starting research paper generation: {research_query}")
        
        # Retrieve similar cases from memory
        similar_cases = self.memory.retrieve_similar_cases(research_query, domain)
        
        # Prepare enhanced query with memory context
        memory_context = ""
        if similar_cases:
            memory_context = "\n\nPrevious research experience context:\n"
            for i, case in enumerate(similar_cases[:2], 1):
                memory_context += f"{i}. Previous research on '{case.query}' in {case.domain}\n"
                memory_context += f"   - Success metrics: {case.success_metrics}\n"
                memory_context += f"   - Key insights: {', '.join(case.insights[:2])}\n"
                memory_context += f"   - Tools used: {', '.join(case.tools_used)}\n"
        
        enhanced_query = f"""
Research Topic: {research_query}
Domain: {domain}
Task: Generate a comprehensive academic research paper

Requirements:
1. Conduct thorough literature review using search tools
2. Extract and analyze relevant papers using document processing
3. Identify research gaps and novel contributions
4. Structure paper with: Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion
5. Include proper citations and references
6. Generate LaTeX format for professional presentation
7. Create any necessary figures, tables, or mathematical formulations

{memory_context}

Please create a detailed, publication-ready research paper.
        """.strip()
        
        try:
            # Use Memento's hierarchical processing
            result = await self.client.process_query(
                query=enhanced_query,
                file=file_path,
                task_id=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Extract tools used from client history
            tools_used = []
            for msg in self.client.shared_history:
                if "tool_calls" in str(msg):
                    # Extract tool names from the conversation
                    content = str(msg.get("content", ""))
                    if "search" in content.lower():
                        tools_used.append("search")
                    if "document" in content.lower():
                        tools_used.append("document_processing")
                    if "code" in content.lower():
                        tools_used.append("code_agent")
            
            tools_used = list(set(tools_used))  # Remove duplicates
            
            # Calculate success metrics
            success_metrics = {
                "completion_score": 1.0 if "FINAL ANSWER:" in result else 0.7,
                "tools_utilization": min(1.0, len(tools_used) / 3.0),
                "content_length": min(1.0, len(result) / 5000.0),
                "academic_quality": 0.8  # Placeholder - could be enhanced with quality assessment
            }
            
            # Extract key insights
            insights = []
            if "research gap" in result.lower():
                insights.append("Identified research gaps")
            if "novel" in result.lower() or "contribution" in result.lower():
                insights.append("Novel contributions identified")
            if "citation" in result.lower() or "reference" in result.lower():
                insights.append("Proper citations included")
            if "latex" in result.lower() or "\\documentclass" in result:
                insights.append("LaTeX formatting applied")
            
            # Store the research case in memory
            research_case = ResearchCase(
                query=research_query,
                domain=domain,
                approach="hierarchical_memento",
                tools_used=tools_used,
                success_metrics=success_metrics,
                insights=insights,
                timestamp=datetime.now(),
                task_complexity=len(enhanced_query.split()),
                final_result=result[:500]  # Store first 500 chars
            )
            
            self.memory.store_case(research_case)
            
            # Save results to files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("memento_research_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # First, determine if result is LaTeX or markdown
            is_latex = "\\documentclass" in result or "\\begin{document}" in result
            
            # Save appropriate format based on content type
            result_file = output_dir / f"research_paper_{timestamp}.md"
            
            if is_latex:
                # If result is LaTeX, convert it to clean markdown for .md file
                clean_markdown = self._convert_latex_to_readable_markdown(result, research_query, domain, timestamp)
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(clean_markdown)
            else:
                # If result is already markdown, save directly
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Research Paper: {research_query}\n\n")
                    f.write(f"**Domain:** {domain}\n")
                    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**Tools Used:** {', '.join(tools_used)}\n\n")
                    f.write("---\n\n")
                    f.write(result)
            
            # Also save as PDF using the converter function
            pdf_file = None
            readable_file = None
            try:
                from convert_research_to_pdf import convert_markdown_to_html_pdf
                html_file = convert_markdown_to_html_pdf(str(result_file))
                if html_file:
                    pdf_file = str(result_file).replace('.md', '_printable.html')
                    logger.info(f"📄 HTML version created for PDF printing: {pdf_file}")
            except Exception as e:
                logger.warning(f"Could not create PDF version: {e}")
            
            # Create additional readable version only if needed (for LaTeX content)
            readable_file = None
            if is_latex:
                try:
                    readable_file = str(result_file).replace('.md', '_readable.md')
                    # Create an even cleaner version for the _readable file
                    extra_clean_markdown = self._convert_latex_to_readable_markdown(result, research_query, domain, timestamp)
                    
                    with open(readable_file, 'w', encoding='utf-8') as f:
                        f.write(extra_clean_markdown)
                    
                    logger.info(f"📖 Extra readable version created: {readable_file}")
                except Exception as e:
                    logger.warning(f"Could not create extra readable version: {e}")
            
            # Always create LaTeX file - convert markdown to LaTeX if needed
            latex_file = output_dir / f"research_paper_{timestamp}.tex"
            pdf_rendered_file = None
            latex_content = None
            
            if "\\documentclass" in result or "\\begin{document}" in result:
                logger.info(f"🔍 LaTeX content detected - using original LaTeX")
                
                # Extract LaTeX content
                latex_start = result.find("\\documentclass")
                if latex_start != -1:
                    latex_content = result[latex_start:]
                    # Find end of LaTeX document
                    latex_end = latex_content.find("\\end{document}")
                    if latex_end != -1:
                        latex_content = latex_content[:latex_end + len("\\end{document}")]
                else:
                    latex_content = result  # Use full result if no clear start found
            else:
                logger.info(f"🔍 No LaTeX content detected - converting markdown to LaTeX")
                # Convert the clean markdown to LaTeX format
                latex_content = self._convert_markdown_to_latex(result, research_query, domain, timestamp)
            
            # Always create LaTeX file
            if latex_content and latex_content.strip():
                with open(latex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                logger.info(f"📊 LaTeX file created: {len(latex_content)} characters")
                
                # Render LaTeX to PDF using tectonic (use existing file)
                logger.info(f"🔍 LaTeX PDF Available: {LATEX_PDF_AVAILABLE}")
                if LATEX_PDF_AVAILABLE:
                    try:
                        logger.info("🚀 Starting LaTeX PDF rendering...")
                        pdf_rendered_file = self._render_latex_pdf_from_file(latex_file)
                        logger.info(f"🎯 PDF rendered from LaTeX: {pdf_rendered_file}")
                    except Exception as e:
                        logger.warning(f"Could not render LaTeX to PDF: {e}")
                        logger.info("💡 Alternative options:")
                        logger.info(f"   📄 LaTeX source: {latex_file}")
                        logger.info(f"   🌐 HTML for PDF: {pdf_file}")
                        logger.info("   🔗 Upload .tex to Overleaf.com for PDF compilation")
                else:
                    logger.info("💡 LaTeX PDF rendering not available - install tectonic for PDF generation")
            else:
                logger.warning("⚠️ Could not create LaTeX content")
                latex_file = None
            
            logger.info(f"✅ Research paper generated successfully!")
            logger.info(f"📄 Clean markdown saved to: {result_file}")
            if latex_file:
                logger.info(f"📊 LaTeX source saved to: {latex_file}")
            if pdf_rendered_file:
                logger.info(f"🎯 LaTeX PDF rendered to: {pdf_rendered_file}")
            if pdf_file:
                logger.info(f"📄 PDF-ready HTML saved to: {pdf_file}")
            
            return {
                "success": True,
                "result": result,
                "research_query": research_query,
                "domain": domain,
                "tools_used": tools_used,
                "success_metrics": success_metrics,
                "insights": insights,
                "result_file": str(result_file),
                "pdf_file": pdf_file,
                "readable_file": readable_file,
                "latex_file": str(latex_file) if latex_file else None,
                "pdf_rendered_file": pdf_rendered_file,
                "memory_cases_used": len(similar_cases)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating research paper: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "research_query": research_query
            }
    
    async def interactive_research_session(self):
        """Interactive research session using Memento"""
        
        if not self.connected:
            await self.initialize()
        
        print("🔬 Memento Interactive Research Session")
        print("=" * 50)
        print("Type 'quit' to exit, 'memory' to view stored cases")
        
        while True:
            try:
                query = input("\n📝 Research topic: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.lower() == 'memory':
                    print(f"\n💾 Memory contains {len(self.memory.cases)} research cases:")
                    for i, case in enumerate(self.memory.cases[-5:], 1):  # Show last 5
                        print(f"{i}. {case.query} ({case.domain}) - Success: {case.success_metrics.get('completion_score', 0):.2f}")
                    continue
                
                if not query:
                    print("❌ Please enter a research topic")
                    continue
                
                domain = input("🏷️  Research domain (default: computer science): ").strip()
                if not domain:
                    domain = "computer science"
                
                print(f"\n🚀 Generating research paper...")
                print("This may take several minutes depending on complexity...")
                
                result = await self.generate_research_paper(query, domain)
                
                if result["success"]:
                    print(f"\n✅ Research paper generated!")
                    print(f"📊 Tools used: {', '.join(result['tools_used'])}")
                    print(f"🎯 Success score: {result['success_metrics']['completion_score']:.2f}")
                    print(f"💡 Insights: {', '.join(result['insights'])}")
                    print(f"\n📄 Generated Files (All 3 Formats):")
                    print(f"   📄 Clean Markdown: {result['result_file']}")
                    if result.get('latex_file'):
                        print(f"   📊 LaTeX Source: {result['latex_file']}")
                    else:
                        print(f"   ⚠️  LaTeX Source: Not created")
                    if result.get('pdf_file'):
                        print(f"   🌐 HTML (for PDF): {result['pdf_file']}")
                    else:
                        print(f"   ⚠️  HTML: Not created")
                    if result.get('pdf_rendered_file'):
                        print(f"   🎯 LaTeX PDF: {result['pdf_rendered_file']}")
                    if result.get('readable_file'):
                        print(f"   📖 Extra Readable: {result['readable_file']}")
                    print(f"\n💡 All formats: .md (readable), .tex (LaTeX), .html (printable)")
                    
                    # Show preview
                    preview = result["result"][:500] + "..." if len(result["result"]) > 500 else result["result"]
                    show_preview = input("\n👀 Show preview? (y/n): ").strip().lower()
                    if show_preview == 'y':
                        print(f"\n📖 Preview:")
                        print("-" * 60)
                        print(preview)
                        print("-" * 60)
                else:
                    print(f"❌ Failed: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    def _convert_latex_to_readable_markdown(self, latex_content: str, title: str, domain: str, timestamp: str) -> str:
        """Convert LaTeX content to clean, readable markdown"""
        
        # Extract content between \begin{document} and \end{document}
        doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if doc_match:
            content = doc_match.group(1)
        else:
            content = latex_content
        
        # Extract title from \title{...}
        title_match = re.search(r'\\title\{([^}]+)\}', latex_content)
        paper_title = title_match.group(1) if title_match else title
        
        # Extract author from \author{...}
        author_match = re.search(r'\\author\{([^}]+)\}', latex_content)
        author = author_match.group(1).replace('\\\\', ', ') if author_match else "Research Agent"
        
        # Start building readable content
        readable = f"# {paper_title}\n\n"
        readable += f"**Authors:** {author}\n"
        readable += f"**Domain:** {domain}\n"
        readable += f"**Generated:** {timestamp}\n\n"
        readable += "---\n\n"
        
        # Convert LaTeX sections to markdown
        content = re.sub(r'\\maketitle\s*', '', content)
        content = re.sub(r'\\section\{([^}]+)\}', r'## \1', content)
        content = re.sub(r'\\subsection\{([^}]+)\}', r'### \1', content)
        content = re.sub(r'\\subsubsection\{([^}]+)\}', r'#### \1', content)
        
        # Convert abstract
        content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', r'## Abstract\n\n\1\n', content, flags=re.DOTALL)
        
        # Convert text formatting
        content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)
        content = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', content)
        content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)
        content = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', content)
        
        # Convert equations and math
        content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'```math\n\1\n```', content, flags=re.DOTALL)
        content = re.sub(r'\\\[(.*?)\\\]', r'```math\n\1\n```', content, flags=re.DOTALL)
        content = re.sub(r'\$([^$]+)\$', r'`\1`', content)
        content = re.sub(r'\\\(([^)]+)\\\)', r'`\1`', content)
        
        # Convert enumerate and itemize
        content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', lambda m: self._convert_enumerate(m.group(1)), content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', lambda m: self._convert_itemize(m.group(1)), content, flags=re.DOTALL)
        
        # Convert tables
        content = re.sub(r'\\begin\{table\}.*?\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}.*?\\end\{table\}', 
                        lambda m: self._convert_table(m.group(1)), content, flags=re.DOTALL)
        
        # Convert figures
        content = re.sub(r'\\begin\{figure\}.*?\\includegraphics\[.*?\]\{([^}]+)\}.*?\\caption\{([^}]+)\}.*?\\end\{figure\}', 
                        r'![Figure: \2](\1)', content, flags=re.DOTALL)
        content = re.sub(r'\\includegraphics\[.*?\]\{([^}]+)\}', r'![Image](\1)', content)
        
        # Convert citations
        content = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', content)
        content = re.sub(r'\\citep\{([^}]+)\}', r'[\1]', content)
        content = re.sub(r'\\citet\{([^}]+)\}', r'[\1]', content)
        
        # Remove figure and table references
        content = re.sub(r'Figure~\\ref\{[^}]+\}', 'Figure', content)
        content = re.sub(r'Table~\\ref\{[^}]+\}', 'Table', content)
        content = re.sub(r'Equation~\\ref\{[^}]+\}', 'Equation', content)
        content = re.sub(r'~\\ref\{[^}]+\}', '', content)
        content = re.sub(r'\\ref\{[^}]+\}', '', content)
        content = re.sub(r'\\label\{[^}]+\}', '', content)
        
        # Remove LaTeX line breaks and spacing
        content = re.sub(r'\\\\', '\n', content)  # Convert LaTeX line breaks
        content = re.sub(r'\\\s*\n', '\n', content)  # Remove backslash at end of line
        content = re.sub(r'\\(?=[^a-zA-Z])', '', content)  # Remove standalone backslashes
        
        # Remove bibliography commands
        content = re.sub(r'\\bibliographystyle\{[^}]+\}', '', content)
        content = re.sub(r'\\bibliography\{[^}]+\}', '', content)
        
        # Remove LaTeX commands we don't handle (more comprehensive)
        content = re.sub(r'\\[a-zA-Z]+\*?\[[^\]]*\]\{[^}]*\}', '', content)  # Commands with optional args
        content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', content)  # Remove remaining LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+\*?', '', content)  # Remove command-only LaTeX
        
        # Clean up remaining braces and special characters
        content = re.sub(r'\{([^{}]*)\}', r'\1', content)  # Remove simple braces
        content = re.sub(r'[{}]', '', content)  # Remove remaining braces
        
        # Clean up whitespace and formatting
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double
        content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)  # Remove leading spaces
        content = re.sub(r'\s+$', '', content, flags=re.MULTILINE)  # Remove trailing spaces
        content = content.strip()
        
        readable += content
        readable += "\n\n---\n\n*Generated by Memento Research Agent with O3 Model*\n"
        readable += "*Original LaTeX version available for academic publication*"
        
        return readable
    
    def _convert_markdown_to_latex(self, markdown_content: str, title: str, domain: str, timestamp: str) -> str:
        """Convert markdown content to LaTeX format"""
        
        # Start with LaTeX document structure
        latex = f"""\\documentclass[11pt]{{article}}
\\usepackage{{amsmath, amssymb, graphicx, url}}
\\usepackage{{natbib}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\geometry{{margin=1in}}

\\title{{{title}}}
\\author{{Author Name \\\\ Institution \\\\ Email}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

"""

        # Process the markdown content
        content = markdown_content.strip()
        
        # Convert markdown headers to LaTeX sections
        content = re.sub(r'^# (.+)$', r'\\section{\1}', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'\\section{\1}', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
        content = re.sub(r'^#### (.+)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
        
        # Convert markdown bold to LaTeX
        content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
        content = re.sub(r'__(.+?)__', r'\\textbf{\1}', content)
        
        # Convert markdown italic to LaTeX
        content = re.sub(r'\*(.+?)\*', r'\\textit{\1}', content)
        content = re.sub(r'_(.+?)_', r'\\textit{\1}', content)
        
        # Convert markdown code to LaTeX
        content = re.sub(r'`(.+?)`', r'\\texttt{\1}', content)
        
        # Convert markdown lists to LaTeX
        # Handle numbered lists
        content = re.sub(r'^(\d+)\.\s+(.+)$', r'\\item \2', content, flags=re.MULTILINE)
        # Handle bullet lists  
        content = re.sub(r'^[-*+]\s+(.+)$', r'\\item \1', content, flags=re.MULTILINE)
        
        # Wrap consecutive items in enumerate/itemize environments
        lines = content.split('\n')
        processed_lines = []
        in_list = False
        list_type = None
        
        for line in lines:
            if line.strip().startswith('\\item'):
                if not in_list:
                    # Determine list type based on original content
                    if any(re.match(r'^\d+\.', orig_line) for orig_line in markdown_content.split('\n')):
                        processed_lines.append('\\begin{enumerate}')
                        list_type = 'enumerate'
                    else:
                        processed_lines.append('\\begin{itemize}')
                        list_type = 'itemize'
                    in_list = True
                processed_lines.append(line)
            else:
                if in_list:
                    processed_lines.append(f'\\end{{{list_type}}}')
                    in_list = False
                    list_type = None
                processed_lines.append(line)
        
        if in_list:  # Close any remaining list
            processed_lines.append(f'\\end{{{list_type}}}')
        
        content = '\n'.join(processed_lines)
        
        # Clean up metadata lines that might be in markdown
        content = re.sub(r'^\*\*Domain:\*\*.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\*\*Generated:\*\*.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\*\*Tools Used:\*\*.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^---\s*$', '', content, flags=re.MULTILINE)
        
        # Remove the title line if it exists (we already have it in LaTeX title)
        content = re.sub(r'^# .*$', '', content, flags=re.MULTILINE)
        
        # Clean up multiple newlines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Add content to LaTeX document
        latex += content.strip()
        
        # Add bibliography if not present
        if '\\bibliography' not in latex:
            latex += """

\\bibliographystyle{plain}
\\bibliography{references}

"""
        
        # Close document
        latex += """\\end{document}"""
        
        return latex
    
    def _render_latex_pdf_from_file(self, latex_file_path: Path) -> str:
        """Render LaTeX to PDF using existing LaTeX file (no duplication)"""
        import subprocess
        import shutil
        
        logger.info("🚀 Starting LaTeX to PDF conversion from existing file...")
        
        if shutil.which("tectonic") is None:
            raise RuntimeError("tectonic is not installed. Install it first on your system.")

        try:
            # Use the existing file's directory and name
            output_dir = latex_file_path.parent
            tex_filename = latex_file_path.name
            pdf_filename = tex_filename.replace('.tex', '.pdf')
            
            # Run tectonic on the existing file
            result = subprocess.run(
                ["tectonic", tex_filename, "--outdir", str(output_dir)],
                cwd=output_dir,
                capture_output=True,
                text=True,
            )

            # Check if PDF was created
            final_pdf = output_dir / pdf_filename
            if not final_pdf.exists():
                raise FileNotFoundError(f"PDF file was not generated. Tectonic output: {result.stderr}")

            logger.info(f"Successfully generated PDF at {final_pdf}")
            return str(final_pdf)

        except Exception as e:
            logger.error(f"Error rendering LaTeX from file: {str(e)}")
            raise
    
    def _convert_enumerate(self, content: str) -> str:
        """Convert LaTeX enumerate to markdown numbered list"""
        items = re.findall(r'\\item\s+([^\\]+?)(?=\\item|$)', content, re.DOTALL)
        result = "\n"
        for i, item in enumerate(items, 1):
            result += f"{i}. {item.strip()}\n"
        return result + "\n"
    
    def _convert_itemize(self, content: str) -> str:
        """Convert LaTeX itemize to markdown bullet list"""
        items = re.findall(r'\\item\s+([^\\]+?)(?=\\item|$)', content, re.DOTALL)
        result = "\n"
        for item in items:
            result += f"- {item.strip()}\n"
        return result + "\n"
    
    def _convert_table(self, content: str) -> str:
        """Convert LaTeX table to markdown table"""
        # This is a simplified table converter
        lines = content.strip().split('\\\\')
        if len(lines) < 2:
            return "\n*Table content*\n\n"
        
        result = "\n"
        for i, line in enumerate(lines):
            if 'toprule' in line or 'midrule' in line or 'bottomrule' in line:
                continue
            
            cells = [cell.strip() for cell in line.split('&')]
            if cells and cells[0]:  # Skip empty lines
                result += "| " + " | ".join(cells) + " |\n"
                if i == 0:  # Add header separator
                    result += "| " + " | ".join(["---"] * len(cells)) + " |\n"
        
        return result + "\n"

    async def cleanup(self):
        """Cleanup resources"""
        if self.connected:
            await self.client.cleanup()

# ============================================================================
# EXAMPLE USAGE AND MAIN FUNCTION
# ============================================================================

async def example_research_queries():
    """Run example research queries"""
    
    agent = MementoResearchAgent()
    
    example_queries = [
        {
            "query": "Transformer architectures for multimodal learning",
            "domain": "machine learning"
        },
        {
            "query": "Blockchain consensus mechanisms for IoT networks", 
            "domain": "distributed systems"
        },
        {
            "query": "Quantum error correction in NISQ devices",
            "domain": "quantum computing"
        }
    ]
    
    print("🔬 Running Example Research Queries with Memento")
    print("=" * 60)
    
    try:
        for i, example in enumerate(example_queries, 1):
            print(f"\n📋 Example {i}/{len(example_queries)}")
            print(f"Query: {example['query']}")
            print(f"Domain: {example['domain']}")
            print("-" * 40)
            
            result = await agent.generate_research_paper(
                example["query"], 
                example["domain"]
            )
            
            if result["success"]:
                print(f"✅ Success! Generated {len(result['result'])} characters")
                print(f"🔧 Tools: {', '.join(result['tools_used'])}")
                print(f"💡 Insights: {len(result['insights'])} key insights found")
            else:
                print(f"❌ Failed: {result['error']}")
            
            # Small delay between queries
            if i < len(example_queries):
                print("⏳ Waiting before next query...")
                await asyncio.sleep(2)
    
    finally:
        await agent.cleanup()

async def main():
    """Main function with options"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set either GROQ_API_KEY or OPENAI_API_KEY in your .env file")
        return
    
    # Display which API is being used
    # Temporarily prioritize OpenAI for O3 testing
    if os.getenv("OPENAI_API_KEY"):
        print("🚀 Using OpenAI API with o3-mini model")
    elif os.getenv("GROQ_API_KEY"):
        print("🚀 Using GROQ API with llama-3.3-70b-versatile model")
    else:
        print("🚀 No API key found")
    
    print("🔬 Memento Library Research Paper Generator")
    print("Using Memento's Hierarchical Agent Architecture")
    print("=" * 60)
    print("\nOptions:")
    print("1. Interactive research session")
    print("2. Run example research queries")
    print("3. Single research query")
    print("4. Exit")
    
    while True:
        choice = input("\n🔢 Choose an option (1-4): ").strip()
        
        if choice == '1':
            agent = MementoResearchAgent()
            try:
                await agent.interactive_research_session()
            finally:
                await agent.cleanup()
            break
            
        elif choice == '2':
            await example_research_queries()
            break
            
        elif choice == '3':
            query = input("\n📝 Enter research query: ").strip()
            if not query:
                print("❌ Please enter a valid query")
                continue
                
            domain = input("🏷️  Research domain (default: computer science): ").strip()
            if not domain:
                domain = "computer science"
            
            agent = MementoResearchAgent()
            try:
                result = await agent.generate_research_paper(query, domain)
                
                if result["success"]:
                    print(f"\n✅ Research paper generated!")
                    print(f"📄 Saved to: {result['result_file']}")
                    if result.get("latex_file"):
                        print(f"📊 LaTeX saved to: {result['latex_file']}")
                else:
                    print(f"❌ Failed: {result['error']}")
            finally:
                await agent.cleanup()
            break
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    asyncio.run(main()) 