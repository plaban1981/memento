# Memento Library Research Agent - Complete Workflow Documentation

## Overview

The `memento_library_research_agent.py` is a sophisticated research paper generation system that leverages Memento's hierarchical agent architecture combined with Model Context Protocol (MCP) tools to create comprehensive academic research papers. This document provides a complete workflow analysis with detailed Mermaid diagrams.

## Architecture Components

### Core Components

1. **MementoResearchAgent**: Main orchestrator class
2. **MementoMemorySystem**: Handles case-based learning and retrieval
3. **HierarchicalClient**: Memento's meta-planner + executor architecture
4. **MCP Tools Integration**: Search, document processing, and other research tools
5. **Multi-format Output System**: Generates `.md`, `.tex`, `.html`, and `.pdf` files

## Complete System Workflow

```mermaid
graph TD
    A[User Input: Research Query] --> B[MementoResearchAgent Initialize]
    B --> C[Load Memory System]
    C --> D[Connect to MCP Servers]
    D --> E[HierarchicalClient Setup]
    E --> F[Meta-Planner + Executor Ready]
    
    F --> G[generate_research_paper]
    G --> H[Retrieve Similar Cases from Memory]
    H --> I[Enhance Query with Memory Context]
    I --> J[Process Query via Hierarchical Client]
    
    J --> K[Meta-Planner: Create Research Plan]
    K --> L[Executor: Execute Research Tasks]
    L --> M[Use MCP Tools]
    M --> N[Search Tool: Literature Review]
    M --> O[Document Tool: Extract Content]
    M --> P[Code Tool: Generate Analysis]
    
    N --> Q[Collect Research Results]
    O --> Q
    P --> Q
    
    Q --> R[Analyze Content Type]
    R --> S{Is LaTeX Content?}
    
    S -->|Yes| T[Extract LaTeX Content]
    S -->|No| U[Use Markdown Content]
    
    T --> V[Convert LaTeX to Clean Markdown for .md]
    U --> W[Format Markdown for .md]
    
    V --> X[Save .md File]
    W --> X
    
    X --> Y[Always Create LaTeX File]
    Y --> Z{Original Content LaTeX?}
    
    Z -->|Yes| AA[Use Original LaTeX]
    Z -->|No| BB[Convert Markdown to LaTeX]
    
    AA --> CC[Save .tex File]
    BB --> CC
    
    CC --> DD[Create HTML for PDF Printing]
    DD --> EE[Attempt LaTeX PDF Rendering]
    EE --> FF{Tectonic Available?}
    
    FF -->|Yes| GG[Render PDF with Tectonic]
    FF -->|No| HH[Skip PDF Rendering]
    
    GG --> II[Store Research Case in Memory]
    HH --> II
    
    II --> JJ[Return Results with File Paths]
    JJ --> KK[Display Results to User]
```

## Detailed Component Architecture

```mermaid
graph TD
    A[MementoResearchAgent] --> B[MementoMemorySystem]
    A --> C[HierarchicalClient]
    A --> D[MCP Server Integration]
    
    B --> B1[ResearchCase Storage]
    B --> B2[Similarity Matching]
    B --> B3[Case Retrieval]
    B --> B4[JSON Persistence]
    
    C --> C1[Meta-Planner LLM]
    C --> C2[Executor LLM]
    C --> C3[Shared History]
    C --> C4[Query Processing]
    
    C1 --> C1A[Research Plan Generation]
    C1 --> C1B[Task Decomposition]
    C1 --> C1C[JSON Response Format]
    
    C2 --> C2A[Task Execution]
    C2 --> C2B[Tool Utilization]
    C2 --> C2C[Content Generation]
    
    D --> D1[Search Tool - SearxNG]
    D --> D2[Document Processing]
    D --> D3[Code Agent]
    D --> D4[Excel Tool]
    D --> D5[Image Tool]
    D --> D6[Math Tool]
    
    D1 --> D1A[Literature Search]
    D1 --> D1B[Web Research]
    D1 --> D1C[Academic Sources]
```

## Multi-Format Output Processing

```mermaid
graph LR
    A[Raw AI Output] --> B{Content Type Detection}
    
    B -->|LaTeX Detected| C[LaTeX Processing Branch]
    B -->|Markdown Detected| D[Markdown Processing Branch]
    
    C --> C1[Extract LaTeX Content]
    C1 --> C2[Save Original LaTeX to .tex]
    C1 --> C3[Convert LaTeX to Clean Markdown]
    C3 --> C4[Save Clean Markdown to .md]
    
    D --> D1[Use Markdown Content]
    D1 --> D2[Convert Markdown to LaTeX]
    D2 --> D3[Save Generated LaTeX to .tex]
    D1 --> D4[Format and Save to .md]
    
    C4 --> E[Create HTML for PDF Printing]
    D4 --> E
    
    C2 --> F{Tectonic Available?}
    D3 --> F
    
    F -->|Yes| G[Render LaTeX to PDF]
    F -->|No| H[Alternative PDF Options]
    
    G --> I[Final Output Files]
    H --> I
    
    I --> I1[research_paper_timestamp.md]
    I --> I2[research_paper_timestamp.tex]
    I --> I3[research_paper_timestamp_printable.html]
    I --> I4[research_paper_timestamp.pdf Optional]
```

## Sequence Diagram: Complete Research Process

```mermaid
sequenceDiagram
    participant U as User
    participant MA as MementoResearchAgent
    participant MS as MementoMemorySystem
    participant HC as HierarchicalClient
    participant MP as Meta-Planner
    participant EX as Executor
    participant MCP as MCP Tools
    participant FS as File System
    
    U->>MA: Research Query + Domain
    MA->>MS: Retrieve Similar Cases
    MS-->>MA: Similar Research Cases
    
    MA->>MA: Enhance Query with Memory Context
    MA->>HC: Process Enhanced Query
    
    HC->>MP: Send Research Query
    MP->>MP: Generate Research Plan
    MP-->>HC: JSON Plan with Tasks
    
    loop For Each Task in Plan
        HC->>EX: Execute Task
        EX->>MCP: Use Research Tools
        MCP-->>EX: Tool Results
        EX-->>HC: Task Results
    end
    
    HC-->>MA: Complete Research Content
    
    MA->>MA: Analyze Content Type
    
    alt LaTeX Content
        MA->>MA: Extract LaTeX Content
        MA->>MA: Convert LaTeX to Markdown
    else Markdown Content
        MA->>MA: Use Markdown Content
        MA->>MA: Convert Markdown to LaTeX
    end
    
    MA->>FS: Save .md File
    MA->>FS: Save .tex File
    MA->>FS: Save .html File
    
    opt If Tectonic Available
        MA->>MA: Render LaTeX to PDF
        MA->>FS: Save .pdf File
    end
    
    MA->>MS: Store Research Case
    MA-->>U: Return Results with File Paths
```

## Core Classes and Functions

### 1. MementoResearchAgent Class

**Purpose**: Main orchestrator for research paper generation

**Key Methods**:
- `__init__()`: Initialize with configurable LLM models (default: o3-mini)
- `initialize()`: Connect to MCP servers and setup tools
- `generate_research_paper()`: Core research generation workflow
- `interactive_research_session()`: Interactive CLI interface
- `cleanup()`: Proper resource cleanup

**Configuration**:
- **meta_model**: Model for planning (default: "o3-mini")
- **exec_model**: Model for execution (default: "o3-mini")
- **server_scripts**: MCP tools to connect (default: search_tool.py)

### 2. MementoMemorySystem Class

**Purpose**: Case-based learning and memory retrieval

**Key Methods**:
- `store_case()`: Store completed research cases
- `retrieve_similar_cases()`: Find relevant past research
- `calculate_similarity()`: Compute case similarity scores
- `load_cases()` / `save_cases()`: Persistence management

**Data Structure**:
```python
@dataclass
class ResearchCase:
    query: str
    domain: str
    approach: str
    tools_used: List[str]
    success_metrics: Dict[str, float]
    insights: List[str]
    timestamp: datetime
    task_complexity: int
    final_result: str
```

### 3. Content Processing Pipeline

**LaTeX to Markdown Conversion**:
- Extract content between `\begin{document}` and `\end{document}`
- Convert sections: `\section{}` → `## Header`
- Convert formatting: `\textbf{}` → `**bold**`
- Handle equations, citations, and references
- Clean up LaTeX commands and syntax

**Markdown to LaTeX Conversion**:
- Add LaTeX document structure
- Convert headers to LaTeX sections
- Handle lists, formatting, and special characters
- Generate proper academic paper structure

## File Output System

The agent **always** generates exactly 4 files for each research paper:

### Core Files (Always Generated)

1. **`.md` File**: Clean, readable markdown
   - Converted from LaTeX if needed
   - Human-readable format
   - Contains all research content

2. **`.tex` File**: LaTeX source code
   - Original LaTeX if AI generated it
   - Converted from markdown if needed
   - Ready for LaTeX compilation

3. **`.html` File**: PDF-ready HTML
   - Styled for browser printing
   - Ctrl+P → Save as PDF workflow
   - Fallback PDF generation method

### Optional File

4. **`.pdf` File**: Compiled PDF (if Tectonic available)
   - Professional publication format
   - Requires Tectonic LaTeX engine
   - Automatic generation when possible

## Memory and Learning System

### Case Storage Format
```json
{
  "query": "Research topic",
  "domain": "Field of study", 
  "approach": "hierarchical_memento",
  "tools_used": ["search", "document_processing"],
  "success_metrics": {
    "completion_score": 0.8,
    "tools_utilization": 0.6,
    "content_length": 0.9,
    "academic_quality": 0.8
  },
  "insights": ["Research gaps identified", "Novel contributions"],
  "timestamp": "2025-01-31T16:40:00",
  "task_complexity": 250,
  "final_result": "First 500 characters of result..."
}
```

### Similarity Matching Algorithm
- **Query similarity**: Cosine similarity of terms
- **Domain matching**: Exact domain match bonus
- **Temporal relevance**: Recent cases weighted higher
- **Success bias**: Higher success scores preferred

## MCP Tools Integration

### Search Tool (SearxNG)
- **Purpose**: Literature search and web research
- **Endpoint**: Local SearxNG instance (localhost:8080)
- **Features**: Privacy-respecting metasearch
- **Usage**: Academic paper discovery, fact verification

### Document Processing Tool
- **Purpose**: Extract and analyze PDFs/documents
- **Features**: Text extraction, content analysis
- **Integration**: Planned for future versions

### Code Agent Tool
- **Purpose**: Generate code, analysis scripts
- **Features**: Data analysis, visualization
- **Integration**: Planned for future versions

## LLM Model Configuration

### Current Setup (O3 Model)
- **Meta-Planner**: Uses OpenAI O3-mini via `responses.create()`
- **Executor**: Uses OpenAI O3-mini via `responses.create()`
- **API**: OpenAI API with special O3 handling
- **Fallback**: GROQ models (llama-3.3-70b-versatile)

### Model Selection Logic
```python
if "o3" in model_name:
    # Use OpenAI API with responses.create()
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    response = await client.responses.create(model=model, input=messages)
else:
    # Use standard chat completions
    response = await client.chat.completions.create(model=model, messages=messages)
```

## Error Handling and Recovery

### Common Issues and Solutions

1. **Missing API Keys**
   - **Error**: HTTP 401 Unauthorized
   - **Solution**: Check `.env` file for `OPENAI_API_KEY` or `GROQ_API_KEY`

2. **SearxNG Not Running**
   - **Error**: Connection refused to localhost:8080
   - **Solution**: Start Docker container with `docker-compose up -d`

3. **Model Not Available**
   - **Error**: HTTP 404 model not found
   - **Solution**: Switch to available model or check API provider

4. **Tectonic PDF Generation Fails**
   - **Error**: Network connectivity issues
   - **Solution**: Use HTML→PDF browser printing fallback

### Graceful Degradation
- PDF generation failures don't stop the process
- Missing tools are logged but don't crash the system
- Memory system continues even if case loading fails
- Alternative output formats provided when primary fails

## Performance Characteristics

### Typical Execution Times
- **Initialization**: 2-5 seconds (MCP server connection)
- **Research Generation**: 30-120 seconds (depends on complexity)
- **File Processing**: 1-3 seconds (format conversions)
- **PDF Rendering**: 5-15 seconds (if Tectonic works)

### Resource Usage
- **Memory**: ~200-500MB (depends on research content)
- **Storage**: ~10-50KB per research paper set
- **Network**: Variable (depends on search tool usage)

## Future Enhancements

### Planned Features
1. **Enhanced Tool Integration**: Document processing, image analysis
2. **Advanced Memory**: Semantic similarity, learning from feedback
3. **Quality Metrics**: Automated paper quality assessment
4. **Collaboration**: Multi-agent research teams
5. **Templates**: Domain-specific paper templates

### Extension Points
- **Custom MCP Tools**: Add domain-specific research tools
- **Model Adapters**: Support additional LLM providers
- **Output Formats**: Add more export formats (Word, PowerPoint)
- **Quality Filters**: Content validation and improvement

## Usage Examples

### Basic Usage
```python
agent = MementoResearchAgent()
await agent.initialize()

result = await agent.generate_research_paper(
    "Mixture of Agents for Enhanced Reasoning",
    domain="artificial intelligence"
)

print(f"Generated: {result['result_file']}")
```

### Custom Model Configuration
```python
agent = MementoResearchAgent(
    meta_model="llama-3.3-70b-versatile",
    exec_model="llama-3.3-70b-versatile"
)
```

### Interactive Session
```python
await agent.interactive_research_session()
# Provides CLI interface for research topics
```

## Conclusion

The Memento Library Research Agent represents a sophisticated integration of:
- **Hierarchical AI Architecture**: Meta-planning + execution
- **Memory-Based Learning**: Case storage and retrieval
- **Tool Integration**: MCP-based research capabilities
- **Multi-Format Output**: Professional document generation
- **Robust Error Handling**: Graceful degradation and recovery

This system demonstrates how modern AI agents can be built with proper architectural patterns, memory systems, and tool integration to create practical, production-ready research automation tools.

