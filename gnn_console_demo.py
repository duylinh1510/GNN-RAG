#!/usr/bin/env python3
"""
Console demo cho GNN-RAG Text2Cypher
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from gnn_rag_demo import Text2CypherGNNRAG

def main():
    print("🧠 Text2Cypher GNN-RAG Console Demo")
    print("=" * 50)
    
    # Initialize system
    print("Khởi tạo GNN-RAG system...")
    try:
        rag_system = Text2CypherGNNRAG()
        print("✅ System initialized successfully!")
        print(f"📊 Graph has {rag_system.graph_data.x.shape[0]} nodes")
        print(f"🔗 Graph has {rag_system.graph_data.edge_index.shape[1]} edges")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("Make sure you have:")
        print("1. Ollama running with llama3.2")
        print("2. All dependencies installed")
        return
    
    print("\n" + "=" * 50)
    print("📝 Example questions:")
    for i, example in enumerate(rag_system.example_queries, 1):
        print(f"{i}. {example['question']}")
    
    print("\n" + "=" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 Enter your question (or 'quit', 'debug', 'help' to see options): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() in ['help', 'h']:
                print("\n📖 Available commands:")
                print("  • Type your question in natural language")
                print("  • 'debug <question>' - Show full prompt sent to LLM") 
                print("  • 'prompt <question>' - Show only the prompt")
                print("  • 'quit' or 'q' - Exit")
                print("  • 'help' or 'h' - Show this help")
                continue
            
            if not question:
                continue
            
            # Check for debug mode
            debug_mode = False
            prompt_only = False
            
            if question.lower().startswith('debug '):
                debug_mode = True
                question = question[6:].strip()  # Remove 'debug ' prefix
            elif question.lower().startswith('prompt '):
                prompt_only = True
                question = question[7:].strip()  # Remove 'prompt ' prefix
            
            print(f"\n🔍 Processing: {question}")
            print("⏳ Retrieving relevant nodes with GNN...")
            
            # Show prompt only if requested
            if prompt_only:
                rag_system.print_prompt(question)
                continue
            
            # Generate cypher
            cypher_query, relevant_nodes = rag_system.generate_cypher(question, debug=debug_mode)
            
            print("\n" + "-" * 40)
            print("🧠 Retrieved Graph Nodes:")
            for node in relevant_nodes:
                print(f"  • {node}")
            
            print("\n" + "-" * 40)
            print("🔧 Generated Context:")
            context = rag_system.retriever.get_context_from_nodes(relevant_nodes)
            print(context)
            
            print("\n" + "-" * 40)
            print("🎯 Generated Cypher Query:")
            print(cypher_query)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main() 