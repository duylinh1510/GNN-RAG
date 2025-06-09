#!/usr/bin/env python3
"""
Simple console version of Text2Cypher RAG Demo
Run this if you prefer console interface over Streamlit
"""

import sys
from text2cypher_demo import Text2CypherRAG

def main():
    print("🔍 Text2Cypher RAG Demo (Console Version)")
    print("=" * 50)
    
    try:
        print("Initializing RAG system...")
        rag_system = Text2CypherRAG()
        print("✅ System initialized successfully!\n")
        
        # Show example queries
        print("💡 Example questions you can ask:")
        for i, example in enumerate(rag_system.example_queries, 1):
            print(f"{i}. {example['question']}")
        print()
        
        while True:
            try:
                question = input("Enter your question (or 'quit' to exit): ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if not question:
                    continue
                
                print("\n🔄 Generating Cypher query...")
                cypher_query = rag_system.generate_cypher(question)
                
                print("\n📋 Generated Cypher Query:")
                print("-" * 40)
                print(cypher_query)
                print("-" * 40)
                
                # Ask if user wants to see context
                show_context = input("\nShow retrieved context? (y/n): ").strip().lower()
                if show_context in ['y', 'yes']:
                    context = rag_system._get_relevant_context(question)
                    print("\n📚 Retrieved Context:")
                    print("-" * 40)
                    print(context)
                    print("-" * 40)
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Install llama3.2 model: ollama pull llama3.2")
        print("3. Install dependencies: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 