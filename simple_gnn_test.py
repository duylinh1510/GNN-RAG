#!/usr/bin/env python3
"""
Simple test để kiểm tra core GNN-RAG functionality
"""

import warnings
warnings.filterwarnings('ignore')

def test_graph_construction():
    """Test basic graph construction"""
    print("🏗️  Testing Graph Construction...")
    
    try:
        from gnn_rag_demo import GraphSchemaBuilder
        
        # Build graph
        builder = GraphSchemaBuilder()
        graph_data, node_to_idx = builder.build_graph()
        
        print(f"✅ Graph created successfully!")
        print(f"   📊 Number of nodes: {graph_data.x.shape[0]}")
        print(f"   🔗 Number of edges: {graph_data.edge_index.shape[1]}")
        print(f"   📏 Feature dimension: {graph_data.x.shape[1]}")
        
        # Show some node examples
        print(f"   🔍 Sample nodes:")
        sample_nodes = list(node_to_idx.keys())[:5]
        for i, node in enumerate(sample_nodes):
            print(f"      {i+1}. {node}")
        
        return True, graph_data, node_to_idx
        
    except Exception as e:
        print(f"❌ Graph construction failed: {e}")
        return False, None, None

def test_gnn_model(graph_data):
    """Test GNN model creation and forward pass"""
    print("\n🧠 Testing GNN Model...")
    
    try:
        from gnn_rag_demo import GNNModel
        
        # Create model
        input_dim = graph_data.x.shape[1]
        model = GNNModel(input_dim=input_dim)
        
        print(f"✅ GNN model created!")
        print(f"   📥 Input dim: {input_dim}")
        print(f"   📤 Output dim: 64")
        print(f"   🔢 Number of layers: 2")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            embeddings = model(graph_data)
        
        print(f"✅ Forward pass successful!")
        print(f"   📊 Output shape: {embeddings.shape}")
        
        return True, model, embeddings
        
    except Exception as e:
        print(f"❌ GNN model test failed: {e}")
        return False, None, None

def test_retrieval(graph_data, node_to_idx, model):
    """Test retrieval functionality"""
    print("\n🔍 Testing Retrieval...")
    
    try:
        from gnn_rag_demo import GNNRetriever
        
        # Create retriever
        retriever = GNNRetriever(graph_data, node_to_idx, model)
        
        print(f"✅ Retriever created!")
        
        # Test query
        test_query = "Show me all products with their categories"
        relevant_nodes = retriever.retrieve_relevant_nodes(test_query, k=3)
        
        print(f"✅ Retrieval successful!")
        print(f"   🤔 Query: {test_query}")
        print(f"   🎯 Retrieved nodes:")
        for i, node in enumerate(relevant_nodes):
            print(f"      {i+1}. {node}")
        
        # Test context generation
        context = retriever.get_context_from_nodes(relevant_nodes)
        print(f"\n   📝 Generated context:")
        print(f"      {context}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple GNN-RAG Functionality Test")
    print("=" * 50)
    
    # Import torch first
    try:
        import torch
        print("✅ PyTorch imported")
    except ImportError:
        print("❌ PyTorch not available")
        exit(1)
    
    # Test 1: Graph Construction
    success, graph_data, node_to_idx = test_graph_construction()
    if not success:
        exit(1)
    
    # Test 2: GNN Model
    success, model, embeddings = test_gnn_model(graph_data)
    if not success:
        exit(1)
    
    # Test 3: Retrieval
    success = test_retrieval(graph_data, node_to_idx, model)
    if not success:
        exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! GNN-RAG is ready to use!")
    print("\n🚀 You can now run:")
    print("   • Console demo: python gnn_console_demo.py")
    print("   • Web demo: streamlit run gnn_rag_demo.py") 