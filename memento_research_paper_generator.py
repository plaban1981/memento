#!/usr/bin/env python3
"""
Memento AI Research Paper Generator Agent
Combines Memento architecture with LangGraph for autonomous research paper generation

Based on:
- Memento architecture (memory-based learning without weight updates)
- LangGraph for multi-agent workflows
- AIwithhassan's ai-researcher patterns (https://github.com/AIwithhassan/ai-researcher)
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import subprocess
import shutil

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Additional imports for research capabilities
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import PyPDF2
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MEMENTO ARCHITECTURE COMPONENTS
# ============================================================================

@dataclass
class ResearchExperience:
    """Represents a research experience in Memento's case bank"""
    query: str
    context: str
    actions_taken: List[str]
    results: Dict[str, Any]
    success_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

class MementoMemoryBank:
    """Memento-style memory bank for storing research experiences"""
    
    def __init__(self, storage_path: str = "memento_case_bank.json"):
        self.storage_path = storage_path
        self.experiences: List[ResearchExperience] = []
        self.load_experiences()
    
    def store_experience(self, experience: ResearchExperience):
        """Store a new research experience"""
        self.experiences.append(experience)
        self.save_experiences()
        logger.info(f"Stored experience for query: {experience.query[:50]}...")
    
    def retrieve_similar_experiences(self, query: str, k: int = 4) -> List[ResearchExperience]:
        """Retrieve K most similar experiences (Memento's optimal K=4)"""
        # Simple similarity based on keyword overlap (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        scored_experiences = []
        for exp in self.experiences:
            exp_words = set(exp.query.lower().split())
            similarity = len(query_words.intersection(exp_words)) / len(query_words.union(exp_words))
            scored_experiences.append((similarity, exp))
        
        # Return top K experiences
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:k]]
    
    def save_experiences(self):
        """Save experiences to disk"""
        data = [asdict(exp) for exp in self.experiences]
        # Convert datetime to string for JSON serialization
        for item in data:
            item['timestamp'] = item['timestamp'].isoformat()
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_experiences(self):
        """Load experiences from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.experiences = []
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    self.experiences.append(ResearchExperience(**item))
                
                logger.info(f"Loaded {len(self.experiences)} experiences from memory bank")
            except Exception as e:
                logger.error(f"Error loading experiences: {e}")
                self.experiences = []

# ============================================================================
# ARXIV RESEARCH TOOLS (Based on AIwithhassan's implementation)
# ============================================================================

def search_arxiv_papers(topic: str, max_results: int = 5) -> dict:
    """Search ArXiv papers using the same logic as AIwithhassan's implementation"""
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

@tool
def arxiv_search(topic: str, max_results: int = 5) -> list[dict]:
    """Search for recently uploaded arXiv papers (from AIwithhassan's implementation)

    Args:
        topic: The topic to search for papers about
        max_results: Maximum number of papers to return (default: 5)

    Returns:
        List of papers with their metadata including title, authors, summary, etc.
    """
    print("ARXIV Agent called")
    print(f"Searching arXiv for papers about: {topic}")
    papers = search_arxiv_papers(topic, max_results)
    if len(papers.get("entries", [])) == 0:
        print(f"No papers found for topic: {topic}")
        raise ValueError(f"No papers found for topic: {topic}")
    print(f"Found {len(papers['entries'])} papers about {topic}")
    return papers["entries"]  # Return the entries list directly

@tool
def read_pdf(url: str) -> str:
    """Read and extract text from a PDF file given its URL (from AIwithhassan's implementation)

    Args:
        url: The URL of the PDF file to read

    Returns:
        The extracted text content from the PDF
    """
    try:
        print(f"📄 Downloading PDF from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        
        print(f"📖 Extracting text from {num_pages} pages...")
        for i, page in enumerate(pdf_reader.pages, 1):
            print(f"   Processing page {i}/{num_pages}")
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text += page_text + "\n\n"

        print(f"✅ Successfully extracted {len(text)} characters of text from PDF")
        return text.strip()
    except Exception as e:
        error_msg = f"Error reading PDF from {url}: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)

@tool
def render_latex_pdf(latex_content: str) -> str:
    """Render a LaTeX document to PDF (from AIwithhassan's implementation)

    Args:
        latex_content: The LaTeX document content as a string

    Returns:
        Path to the generated PDF document
    """
    if shutil.which("tectonic") is None:
        raise RuntimeError(
            "tectonic is not installed. Install it first on your system."
        )

    try:
        # Create directory
        output_dir = Path("output").absolute()
        output_dir.mkdir(exist_ok=True)
        
        # Setup filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tex_filename = f"paper_{timestamp}.tex"
        pdf_filename = f"paper_{timestamp}.pdf"
        
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

@tool
def analyze_arxiv_paper(paper_data: dict, research_topic: str) -> str:
    """Analyze an ArXiv paper for relevance and extract key insights
    
    Args:
        paper_data: Dictionary containing paper metadata (title, authors, summary, pdf)
        research_topic: The research topic to assess relevance against
        
    Returns:
        Analysis of the paper including methodology, contributions, and relevance
    """
    try:
        analysis = {
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "summary": paper_data.get("summary", ""),
            "pdf_url": paper_data.get("pdf", ""),
            "categories": paper_data.get("categories", []),
            "full_content": "",
            "methodology": "",
            "key_contributions": [],
            "limitations": [],
            "relevance_score": 0.0,
            "key_insights": []
        }
        
        # Try to read the full PDF content
        if analysis["pdf_url"]:
            try:
                print(f"📄 Reading full content of: {analysis['title'][:60]}...")
                full_content = read_pdf(analysis["pdf_url"])
                analysis["full_content"] = full_content[:5000]  # First 5000 chars for analysis
                
                # Extract methodology and contributions using simple text analysis
                content_lower = full_content.lower()
                
                # Look for methodology indicators
                method_indicators = ["method", "approach", "algorithm", "technique", "framework"]
                for indicator in method_indicators:
                    if indicator in content_lower:
                        # Find sentences containing methodology keywords
                        sentences = full_content.split('.')
                        method_sentences = [s.strip() for s in sentences if indicator in s.lower()]
                        if method_sentences:
                            analysis["methodology"] = method_sentences[0][:200] + "..."
                            break
                
                # Look for contribution indicators
                contrib_indicators = ["contribution", "novel", "propose", "introduce", "present"]
                contributions = []
                for indicator in contrib_indicators:
                    if indicator in content_lower:
                        sentences = full_content.split('.')
                        contrib_sentences = [s.strip() for s in sentences if indicator in s.lower()]
                        contributions.extend(contrib_sentences[:2])
                
                analysis["key_contributions"] = [c[:150] + "..." for c in contributions[:3]]
                
                # Calculate relevance score based on keyword overlap
                topic_words = set(research_topic.lower().split())
                content_words = set(content_lower.split())
                common_words = topic_words.intersection(content_words)
                analysis["relevance_score"] = min(1.0, len(common_words) / len(topic_words))
                
                print(f"✅ Analysis complete. Relevance score: {analysis['relevance_score']:.2f}")
                
            except Exception as e:
                print(f"⚠️ Could not read PDF content: {e}")
                # Fallback to summary analysis
                summary_lower = analysis["summary"].lower()
                topic_words = set(research_topic.lower().split())
                summary_words = set(summary_lower.split())
                common_words = topic_words.intersection(summary_words)
                analysis["relevance_score"] = min(1.0, len(common_words) / len(topic_words))
        
        # Generate key insights
        if analysis["relevance_score"] > 0.3:
            analysis["key_insights"].append("High relevance to research topic")
        if analysis["methodology"]:
            analysis["key_insights"].append("Clear methodology identified")
        if analysis["key_contributions"]:
            analysis["key_insights"].append("Novel contributions found")
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        error_msg = f"Error analyzing ArXiv paper: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg})

# Additional research tools
@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    try:
        search_engine = DuckDuckGoSearchRun()
        results = search_engine.run(query)
        return f"Web search results for '{query}':\n{results}"
    except Exception as e:
        return f"Error in web search: {str(e)}"

@tool
def extract_keywords(text: str) -> str:
    """Extract key research topics from text using AI"""
    try:
        # Use environment variable to determine which model to use
        if os.getenv("GOOGLE_API_KEY"):
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = ChatPromptTemplate.from_template("""
        Extract the top 5 most important research keywords/topics from this text.
        Return them as a JSON list of strings.
        
        Text: {text}
        
        Keywords:
        """)
        
        response = llm.invoke(prompt.format(text=text))
        return response.content
    except Exception as e:
        return f"Error extracting keywords: {str(e)}"

# ============================================================================
# LANGGRAPH STATE (Based on AIwithhassan's State structure)
# ============================================================================

class State(TypedDict):
    """State for the research workflow (based on AIwithhassan's implementation)"""
    messages: Annotated[list, add_messages]
    research_query: str
    research_plan: str
    collected_papers: List[Dict[str, Any]]
    paper_contents: List[str]
    analysis: str
    paper_outline: str
    latex_content: str
    final_pdf_path: str
    similar_experiences: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    success_score: float

# ============================================================================
# MEMENTO RESEARCH AGENT WITH LANGGRAPH
# ============================================================================

class MementoResearchAgent:
    """Main research agent using Memento architecture with LangGraph"""
    
    def __init__(self, openai_api_key: str = None, google_api_key: str = None):
        self.memory_bank = MementoMemoryBank()
        
        # Set up model (prefer Google Gemini if available, fallback to OpenAI)
        if google_api_key or os.getenv("GOOGLE_API_KEY"):
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                api_key=google_api_key or os.getenv("GOOGLE_API_KEY")
            )
            logger.info("Using Google Gemini model")
        elif openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
            logger.info("Using OpenAI GPT-4o-mini model")
        else:
            raise ValueError("Please provide either OPENAI_API_KEY or GOOGLE_API_KEY")
        
        # Define tools (based on AIwithhassan's tools + enhanced ArXiv analysis)
        self.tools = [arxiv_search, read_pdf, analyze_arxiv_paper, render_latex_pdf, web_search, extract_keywords]
        self.tool_node = ToolNode(self.tools)
        
        # Bind tools to model
        self.model = self.model.bind_tools(self.tools)
        
        # Checkpointer for conversation memory (initialize before graph)
        self.checkpointer = MemorySaver()
        
        # Initialize LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow (based on AIwithhassan's structure)"""
        
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes (agents)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("memory_retriever", self.memory_retriever)
        workflow.add_node("experience_updater", self.experience_updater)
        
        # Define the workflow (similar to AIwithhassan's structure)
        workflow.set_entry_point("memory_retriever")
        workflow.add_edge("memory_retriever", "agent")
        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_edge("tools", "agent")
        workflow.add_edge("agent", "experience_updater")
        workflow.add_edge("experience_updater", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def call_model(self, state: State):
        """Call the model (based on AIwithhassan's call_model)"""
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: State) -> Literal["tools", "experience_updater"]:
        """Determine next step (based on AIwithhassan's should_continue)"""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "experience_updater"
    
    def memory_retriever(self, state: State) -> State:
        """Retrieve similar experiences from Memento memory bank"""
        logger.info("🗃️ Memory Retriever: Searching for similar research experiences")
        
        # Extract query from messages
        if state.get("research_query"):
            query = state["research_query"]
        else:
            # Extract from last user message
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            query = user_messages[-1].content if user_messages else "general research"
            state["research_query"] = query
        
        similar_experiences = self.memory_bank.retrieve_similar_experiences(query, k=4)
        
        # Convert to serializable format
        state["similar_experiences"] = [
            {
                "query": exp.query,
                "actions_taken": exp.actions_taken,
                "success_score": exp.success_score,
                "key_insights": exp.metadata.get("key_insights", [])
            }
            for exp in similar_experiences
        ]
        
        # Add memory context to messages if we have similar experiences
        if similar_experiences:
            memory_context = "Based on similar past research experiences:\n"
            for exp in similar_experiences[:2]:  # Top 2 experiences
                memory_context += f"- Previous research on '{exp.query}' achieved {exp.success_score:.2f} success\n"
                memory_context += f"  Key insights: {exp.metadata.get('key_insights', ['No insights'])}\n"
            
            state["messages"].append(SystemMessage(content=memory_context))
        
        state["current_step"] = "memory_retrieved"
        return state
    
    def experience_updater(self, state: State) -> State:
        """Update Memento memory bank with new experience"""
        logger.info("💾 Experience Updater: Storing research experience")
        
        # Extract information from the conversation
        query = state.get("research_query", "Unknown query")
        actions_taken = []
        
        # Analyze messages to determine actions taken and ArXiv usage
        arxiv_papers_found = 0
        arxiv_papers_analyzed = 0
        
        for msg in state["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('args', {})
                    actions_taken.append(f"{tool_name}: {tool_args}")
                    
                    # Track ArXiv-specific metrics
                    if tool_name == 'arxiv_search':
                        # Count papers found (estimate from typical ArXiv search)
                        arxiv_papers_found += tool_args.get('max_results', 5)
                    elif tool_name == 'analyze_arxiv_paper':
                        arxiv_papers_analyzed += 1
        
        # Calculate success score based on conversation quality, tool usage, and ArXiv integration
        base_score = min(1.0, len(actions_taken) / 4.0)  # Base score from tool usage
        arxiv_bonus = 0.2 if arxiv_papers_found > 0 else 0  # Bonus for using ArXiv
        analysis_bonus = 0.1 if arxiv_papers_analyzed > 0 else 0  # Bonus for paper analysis
        success_score = min(1.0, base_score + arxiv_bonus + analysis_bonus)
        
        # Extract key insights from the conversation
        ai_messages = [msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
        key_insights = ai_messages[-1][:200] if ai_messages else "No insights"
        
        # Create new experience
        experience = ResearchExperience(
            query=query,
            context="Research paper generation session",
            actions_taken=actions_taken,
            results={
                "tools_used": len(actions_taken),
                "conversation_length": len(state["messages"]),
                "pdf_generated": state.get("final_pdf_path") is not None
            },
            success_score=success_score,
            timestamp=datetime.now(),
            metadata={
                "key_insights": key_insights,
                "tools_used": actions_taken,
                "arxiv_papers_found": arxiv_papers_found,
                "arxiv_papers_analyzed": arxiv_papers_analyzed,
                "research_quality_score": success_score
            }
        )
        
        # Store in memory bank
        self.memory_bank.store_experience(experience)
        
        state["success_score"] = success_score
        state["current_step"] = "complete"
        
        return state
    
    def print_stream(self, stream):
        """Print stream output (from AIwithhassan's implementation)"""
        for s in stream:
            message = s["messages"][-1]
            print(f"Message received: {message.content[:200]}...")
            if hasattr(message, 'pretty_print'):
                message.pretty_print()
            else:
                print(f"Agent: {message.content}")
    
    async def generate_research_paper(self, research_topic: str, interactive: bool = False) -> Dict[str, Any]:
        """Main method to generate a research paper"""
        logger.info(f"🚀 Starting research paper generation for: {research_topic}")
        
        # Initial prompt (based on AIwithhassan's INITIAL_PROMPT)
        INITIAL_PROMPT = """
        You are an expert researcher in the fields of physics, mathematics,
        computer science, quantitative biology, quantitative finance, statistics,
        electrical engineering and systems science, and economics.

        You are going to analyze recent research papers in one of these fields in
        order to identify promising new research directions and then write a new
        research paper. For research information or getting papers, ALWAYS use arxiv.org.
        You will use the tools provided to search for papers, read them, and write a new
        paper based on the ideas you find.

        Focus on the research topic: {research_topic}

        Enhanced ArXiv Research Steps:
        1. Search for recent papers on this topic using arxiv_search (get 8-10 papers)
        2. For each relevant paper, use analyze_arxiv_paper to extract:
           - Key methodology and approach
           - Main contributions and novelty
           - Limitations and gaps
           - Relevance score to your research topic
        3. Read the most promising papers' full content using read_pdf
        4. Identify research gaps and opportunities from the literature analysis
        5. Propose novel contributions that address identified gaps
        6. Write a comprehensive research paper with:
           - Literature review citing ArXiv papers with proper IDs
           - Clear methodology section
           - Mathematical equations and formulations
           - Results and analysis
           - Discussion of limitations and future work
        7. Render the final paper as a LaTeX PDF using render_latex_pdf

        Important Guidelines:
        - Always cite ArXiv papers with their full ArXiv IDs (e.g., arXiv:2301.12345)
        - Include PDF links in your references
        - Focus on recent papers (2022-2024) for current state-of-the-art
        - Ensure LaTeX code is error-free for successful PDF generation
        - Build upon existing work rather than duplicating it
        """
        
        # Initialize state
        initial_state = State(
            messages=[
                SystemMessage(content=INITIAL_PROMPT.format(research_topic=research_topic)),
                HumanMessage(content=f"Please research and write a comprehensive paper on: {research_topic}")
            ],
            research_query=research_topic,
            research_plan="",
            collected_papers=[],
            paper_contents=[],
            analysis="",
            paper_outline="",
            latex_content="",
            final_pdf_path="",
            similar_experiences=[],
            current_step="initialized",
            iteration_count=0,
            success_score=0.0
        )
        
        # Configuration
        config = {"configurable": {"thread_id": f"research_{int(time.time())}"}}
        
        try:
            if interactive:
                # Interactive mode (like AIwithhassan's while loop)
                print("🔬 Interactive Research Mode - Type 'quit' to exit")
                
                current_state = initial_state
                while True:
                    user_input = input("\nUser: ").strip()
                    if user_input.lower() == 'quit':
                        break
                    
                    if user_input:
                        current_state["messages"].append(HumanMessage(content=user_input))
                        
                        print("\n🤖 Agent Response:")
                        stream = self.graph.stream(current_state, config, stream_mode="values")
                        self.print_stream(stream)
                        
                        # Update state with latest response
                        for s in stream:
                            current_state = s
            else:
                # Autonomous mode
                final_state = self.graph.invoke(initial_state, config)
                
                # Extract results
                ai_messages = [msg.content for msg in final_state["messages"] if isinstance(msg, AIMessage)]
                
                # Save results to files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("research_outputs")
                output_dir.mkdir(exist_ok=True)
                
                # Save conversation log
                conversation_file = output_dir / f"research_conversation_{timestamp}.md"
                with open(conversation_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Research Session: {research_topic}\n\n")
                    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("## Conversation Log\n\n")
                    
                    for msg in final_state["messages"]:
                        if isinstance(msg, HumanMessage):
                            f.write(f"**User:** {msg.content}\n\n")
                        elif isinstance(msg, AIMessage):
                            f.write(f"**Agent:** {msg.content}\n\n")
                        elif isinstance(msg, SystemMessage):
                            f.write(f"**System:** {msg.content}\n\n")
                
                # Check if LaTeX PDF was generated
                pdf_path = final_state.get("final_pdf_path", "")
                if not pdf_path:
                    # Check if any AI message contains LaTeX content
                    for msg in ai_messages:
                        if "\\documentclass" in msg or "\\begin{document}" in msg:
                            try:
                                pdf_path = render_latex_pdf(msg)
                                break
                            except Exception as e:
                                logger.warning(f"Failed to render LaTeX: {e}")
                
                logger.info(f"✅ Research session completed!")
                logger.info(f"📄 Conversation saved to: {conversation_file}")
                if pdf_path:
                    logger.info(f"📊 PDF generated at: {pdf_path}")
                
                return {
                    "success": True,
                    "conversation_content": "\n".join(ai_messages),
                    "research_topic": research_topic,
                    "conversation_file": str(conversation_file),
                    "pdf_path": pdf_path,
                    "success_score": final_state.get("success_score", 0.0),
                    "messages_count": len(final_state["messages"])
                }
                
        except Exception as e:
            logger.error(f"❌ Error in research session: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# MAIN EXECUTION (Based on AIwithhassan's structure)
# ============================================================================

async def main():
    """Main execution function"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not openai_api_key and not google_api_key:
        print("❌ Please set either OPENAI_API_KEY or GOOGLE_API_KEY in your .env file")
        return
    
    # Initialize the research agent
    agent = MementoResearchAgent(openai_api_key, google_api_key)
    
    # Example research topics (based on AIwithhassan's domain focus)
    example_topics = [
        "Recent advances in transformer architectures for natural language processing",
        "Applications of quantum machine learning in optimization problems",
        "Federated learning approaches for privacy-preserving AI in healthcare",
        "Graph neural networks for molecular property prediction",
        "Reinforcement learning for autonomous vehicle navigation systems"
    ]
    
    print("🔬 Memento AI Research Paper Generator")
    print("Based on Memento Architecture + LangGraph + AIwithhassan's patterns")
    print("=" * 70)
    print("\nExample research topics:")
    for i, topic in enumerate(example_topics, 1):
        print(f"{i}. {topic}")
    
    print("\nOptions:")
    print("- Enter a number (1-5) to use an example topic")
    print("- Enter your own research topic")
    print("- Type 'interactive' for interactive research mode")
    print("- Type 'quit' to exit")
    
    while True:
        user_input = input("\n🔍 Enter your choice: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'interactive':
            print("\n🔄 Starting interactive research mode...")
            print("The agent will guide you through the research process step by step.")
            
            topic = input("\n📝 Enter your research topic: ").strip()
            if not topic:
                print("❌ Please enter a valid research topic")
                continue
            
            result = await agent.generate_research_paper(topic, interactive=True)
            
        elif user_input.isdigit() and 1 <= int(user_input) <= len(example_topics):
            topic = example_topics[int(user_input) - 1]
            print(f"\n🚀 Researching: {topic}")
            print("This may take several minutes...")
            
            result = await agent.generate_research_paper(topic, interactive=False)
            
            if result["success"]:
                print(f"\n✅ Research completed successfully!")
                print(f"📄 Conversation saved to: {result['conversation_file']}")
                if result.get("pdf_path"):
                    print(f"📊 PDF generated at: {result['pdf_path']}")
                print(f"🎯 Success score: {result['success_score']:.2f}")
                print(f"💬 Total messages: {result['messages_count']}")
            else:
                print(f"❌ Research failed: {result['error']}")
        
        else:
            # Custom topic
            topic = user_input
            if not topic:
                print("❌ Please enter a valid research topic")
                continue
            
            print(f"\n🚀 Researching: {topic}")
            print("This may take several minutes...")
            
            result = await agent.generate_research_paper(topic, interactive=False)
            
            if result["success"]:
                print(f"\n✅ Research completed successfully!")
                print(f"📄 Conversation saved to: {result['conversation_file']}")
                if result.get("pdf_path"):
                    print(f"📊 PDF generated at: {result['pdf_path']}")
                print(f"🎯 Success score: {result['success_score']:.2f}")
            else:
                print(f"❌ Research failed: {result['error']}")
        
        continue_research = input("\n🔄 Generate another research paper? (y/n): ").strip().lower()
        if continue_research != 'y':
            break
    
    print("\n👋 Thank you for using Memento AI Research Paper Generator!")
    print("💾 All research experiences have been stored in the memory bank for future use.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 