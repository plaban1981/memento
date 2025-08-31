#!/usr/bin/env python3
"""
Example Usage of Memento AI Research Paper Generator
Demonstrates how to use the research agent programmatically
"""

import asyncio
import os
from dotenv import load_dotenv
from memento_research_paper_generator import MementoResearchAgent

async def example_research_session():
    """Example research session"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OPENAI_API_KEY in a .env file")
        return
    
    # Initialize the research agent
    print("🚀 Initializing Memento Research Agent...")
    agent = MementoResearchAgent(openai_api_key)
    
    # Example research queries
    research_queries = [
        "Applications of large language models in code generation",
        "Sustainable AI: Energy-efficient machine learning approaches",
        "Federated learning for privacy-preserving healthcare AI"
    ]
    
    print("\n🔬 Starting Research Sessions")
    print("=" * 50)
    
    for i, query in enumerate(research_queries, 1):
        print(f"\n📋 Research Session {i}/3")
        print(f"Query: {query}")
        print("-" * 50)
        
        # Generate research paper
        result = await agent.generate_research_paper(query)
        
        if result["success"]:
            print(f"✅ Success! Generated paper with {result['data_sources']} sources")
            print(f"🎯 Success Score: {result['success_score']:.2f}")
            print(f"📄 Paper saved to: {result['paper_file']}")
            
            # Show a brief preview
            preview = result["paper_content"][:300] + "..."
            print(f"\n📖 Preview:\n{preview}")
            
        else:
            print(f"❌ Failed: {result['error']}")
        
        # Add delay between requests
        if i < len(research_queries):
            print("\n⏳ Waiting before next research session...")
            await asyncio.sleep(2)
    
    print(f"\n🎉 Completed all research sessions!")
    print(f"💾 Memory bank now contains experiences for future research")

async def interactive_research():
    """Interactive research session"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set your OPENAI_API_KEY in a .env file")
        return
    
    # Initialize the research agent
    print("🚀 Initializing Memento Research Agent...")
    agent = MementoResearchAgent(openai_api_key)
    
    print("\n🔍 Interactive Research Mode")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\n📝 Enter your research query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            print("❌ Please enter a valid query")
            continue
        
        print(f"\n🚀 Researching: {query}")
        print("This may take a few minutes...")
        
        result = await agent.generate_research_paper(query)
        
        if result["success"]:
            print(f"\n✅ Research completed successfully!")
            print(f"📊 Sources: {result['data_sources']}")
            print(f"🎯 Score: {result['success_score']:.2f}")
            print(f"📄 Paper: {result['paper_file']}")
            
            # Ask if user wants to see preview
            show_preview = input("\n👀 Show paper preview? (y/n): ").strip().lower()
            if show_preview == 'y':
                print(f"\n📖 Paper Preview:")
                print("-" * 60)
                print(result["paper_content"][:800] + "...")
                print("-" * 60)
        else:
            print(f"❌ Research failed: {result['error']}")

def main():
    """Main function with options"""
    
    print("🔬 Memento AI Research Paper Generator - Examples")
    print("=" * 55)
    print("\nChoose an option:")
    print("1. Run example research sessions (3 predefined queries)")
    print("2. Interactive research mode")
    print("3. Exit")
    
    while True:
        choice = input("\n🔢 Enter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 Running example research sessions...")
            asyncio.run(example_research_session())
            break
        elif choice == '2':
            print("\n🔍 Starting interactive research mode...")
            asyncio.run(interactive_research())
            break
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 