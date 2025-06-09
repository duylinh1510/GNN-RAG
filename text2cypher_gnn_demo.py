import os
from typing import Dict, List, Any, Tuple
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

class GraphSchemaBuilder:
    """Xây dựng graph từ schema database"""
    
    def __init__(self):
        self.node_types = ['Product', 'Category', 'Supplier', 'Customer', 'Order']
        self.relationships = [
            ('Product', 'PART_OF', 'Category'),
            ('Supplier', 'SUPPLIES', 'Product'), 
            ('Customer', 'PURCHASED', 'Order'),
            ('Order', 'ORDERS', 'Product')
        ]
        self.properties = self._load_properties()
        
    def _load_properties(self) -> Dict[str, List[str]]:
        """Load properties cho mỗi node type từ schema"""
        return {
            'Product': ['productName', 'quantityPerUnit', 'unitsOnOrder', 'supplierID', 
                       'productID', 'discontinued', 'categoryID', 'reorderLevel', 
                       'unitsInStock', 'unitPrice'],
            'Category': ['picture', 'categoryID', 'description', 'categoryName'],
            'Supplier': ['companyName', 'contactName', 'homePage', 'phone', 'postalCode',
                        'contactTitle', 'region', 'address', 'fax', 'supplierID', 
                        'country', 'city'],
            'Customer': ['fax', 'companyName', 'customerID', 'phone', 'contactName',
                        'contactTitle', 'region', 'address', 'postalCode', 'country', 'city'],
            'Order': ['shipName', 'requiredDate', 'shipCity', 'employeeID', 'shipPostalCode',
                     'shippedDate', 'freight', 'orderDate', 'orderID', 'shipAddress',
                     'customerID', 'shipCountry', 'shipVia', 'shipRegion']
        }
    
    def build_graph(self) -> Tuple[Data, Dict]:
        """Xây dựng PyTorch Geometric graph"""
        
        # Tạo nodes - mỗi node type và property sẽ là một node
        nodes = []
        node_to_idx = {}
        node_features = []
        
        # Encoder cho text
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        idx = 0
        
        # Add node types
        for node_type in self.node_types:
            nodes.append(f"NodeType:{node_type}")
            node_to_idx[f"NodeType:{node_type}"] = idx
            # Encode tên node type
            embedding = encoder.encode(f"Database node type {node_type}")
            node_features.append(embedding)
            idx += 1
        
        # Add properties for each node type
        for node_type, props in self.properties.items():
            for prop in props:
                node_name = f"Property:{node_type}.{prop}"
                nodes.append(node_name)
                node_to_idx[node_name] = idx
                # Encode property name with context
                embedding = encoder.encode(f"{node_type} has property {prop}")
                node_features.append(embedding)
                idx += 1
        
        # Add relationships
        for src, rel, dst in self.relationships:
            rel_name = f"Relationship:{src}-{rel}->{dst}"
            nodes.append(rel_name)
            node_to_idx[rel_name] = idx
            # Encode relationship
            embedding = encoder.encode(f"{src} has relationship {rel} to {dst}")
            node_features.append(embedding)
            idx += 1
        
        # Tạo edges
        edge_index = []
        
        # Connect node types to their properties
        for node_type, props in self.properties.items():
            node_type_idx = node_to_idx[f"NodeType:{node_type}"]
            for prop in props:
                prop_idx = node_to_idx[f"Property:{node_type}.{prop}"]
                # Bidirectional edges
                edge_index.append([node_type_idx, prop_idx])
                edge_index.append([prop_idx, node_type_idx])
        
        # Connect relationships to node types
        for src, rel, dst in self.relationships:
            src_idx = node_to_idx[f"NodeType:{src}"]
            dst_idx = node_to_idx[f"NodeType:{dst}"]
            rel_idx = node_to_idx[f"Relationship:{src}-{rel}->{dst}"]
            
            # Connect relationship to source and destination
            edge_index.append([src_idx, rel_idx])
            edge_index.append([rel_idx, dst_idx])
            edge_index.append([rel_idx, src_idx])
            edge_index.append([dst_idx, rel_idx])
        
        # Convert to tensors
        node_features = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Tạo PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index)
        
        return data, node_to_idx

class GNNModel(nn.Module):
    """Simple GNN model để tạo node embeddings"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 2):
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

class GNNRetriever:
    """Retriever sử dụng GNN embeddings thay vì FAISS"""
    
    def __init__(self, graph_data: Data, node_to_idx: Dict, model: GNNModel):
        self.graph_data = graph_data
        self.node_to_idx = node_to_idx
        self.model = model
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Generate embeddings cho tất cả nodes
        self.model.eval()
        with torch.no_grad():
            self.node_embeddings = self.model(graph_data)
    
    def retrieve_relevant_nodes(self, query: str, k: int = 5) -> List[str]:
        """Retrieve k nodes relevants nhất cho query"""
        
        # Encode query
        query_embedding = self.encoder.encode(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float)
        
        # Tính similarity với tất cả node embeddings
        similarities = F.cosine_similarity(
            query_tensor.unsqueeze(0), 
            self.node_embeddings
        )
        
        # Get top k indices
        top_k_indices = similarities.argsort(descending=True)[:k]
        
        # Convert indices back to node names
        idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        relevant_nodes = [idx_to_node[idx.item()] for idx in top_k_indices]
        
        return relevant_nodes
    
    def get_context_from_nodes(self, relevant_nodes: List[str]) -> str:
        """Tạo context string từ relevant nodes"""
        context_parts = []
        
        for node in relevant_nodes:
            if node.startswith("NodeType:"):
                node_type = node.split(":")[1]
                context_parts.append(f"Node Type: {node_type}")
                
            elif node.startswith("Property:"):
                prop_info = node.split(":")[1]
                context_parts.append(f"Property: {prop_info}")
                
            elif node.startswith("Relationship:"):
                rel_info = node.split(":")[1]
                context_parts.append(f"Relationship: {rel_info}")
        
        return "\n".join(context_parts)

class Text2CypherGNNRAG:
    """Main class implementing Text2Cypher với GNN-RAG"""
    
    def __init__(self):
        # Initialize Ollama
        self.llm = OllamaLLM(model="llama3.2")
        
        # Build graph từ schema
        self.graph_builder = GraphSchemaBuilder()
        self.graph_data, self.node_to_idx = self.graph_builder.build_graph()
        
        # Initialize GNN model
        input_dim = self.graph_data.x.shape[1]  # Dimension của sentence transformer
        self.gnn_model = GNNModel(input_dim=input_dim)
        
        # Train model đơn giản (unsupervised)
        self._train_gnn()
        
        # Initialize retriever
        self.retriever = GNNRetriever(self.graph_data, self.node_to_idx, self.gnn_model)
        
        # Example queries
        self.example_queries = [
            {
                "question": "Show me all products with their categories",
                "cypher": "MATCH (p:Product)-[:PART_OF]->(c:Category) RETURN p.productName, c.categoryName"
            },
            {
                "question": "Find suppliers from UK", 
                "cypher": "MATCH (s:Supplier) WHERE s.country = 'UK' RETURN s.companyName, s.city"
            },
            {
                "question": "Get orders with customer information",
                "cypher": "MATCH (c:Customer)-[:PURCHASED]->(o:Order) RETURN c.companyName, o.orderID, o.orderDate"
            }
        ]
    
    def _train_gnn(self):
        """Train GNN model đơn giản để học structure"""
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        self.gnn_model.train()
        for epoch in range(50):  # Simple training
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gnn_model(self.graph_data)
            
            # Simple reconstruction loss (nodes connected by edges should be similar)
            edge_index = self.graph_data.edge_index
            src_embeddings = embeddings[edge_index[0]]
            dst_embeddings = embeddings[edge_index[1]]
            
            # Cosine similarity loss
            similarities = F.cosine_similarity(src_embeddings, dst_embeddings)
            loss = -similarities.mean()  # Maximize similarity
            
            loss.backward()
            optimizer.step()
    
    def generate_cypher(self, question: str) -> Tuple[str, List[str]]:
        """Generate Cypher query từ natural language question"""
        
        # Retrieve relevant nodes using GNN
        relevant_nodes = self.retriever.retrieve_relevant_nodes(question, k=5)
        context = self.retriever.get_context_from_nodes(relevant_nodes)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert in converting natural language questions to Neo4j Cypher queries.

Given the following database schema context retrieved using Graph Neural Networks:
{context}

The database has these relationships:
- (:Product)-[:PART_OF]->(:Category)
- (:Supplier)-[:SUPPLIES]->(:Product)  
- (:Customer)-[:PURCHASED]->(:Order)
- (:Order)-[:ORDERS]->(:Product)

Convert this natural language question to a Cypher query:
Question: {question}

Rules:
1. Only return the Cypher query, no explanations
2. Use proper Neo4j syntax
3. Use the exact node labels and property names from the schema
4. Use appropriate relationships from the schema
5. Include necessary WHERE clauses for filtering
6. Use RETURN to specify what data to retrieve

Cypher Query:
"""
        )
        
        # Generate query
        prompt = prompt_template.format(context=context, question=question)
        cypher_query = self.llm.invoke(prompt)
        
        # Clean up response
        cypher_query = cypher_query.strip()
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        return cypher_query, relevant_nodes
    
    def visualize_graph(self):
        """Tạo visualization của graph schema"""
        G = nx.Graph()
        
        # Add nodes
        for node_name in self.node_to_idx.keys():
            if node_name.startswith("NodeType:"):
                G.add_node(node_name, node_type="entity")
            elif node_name.startswith("Property:"):
                G.add_node(node_name, node_type="property")
            else:
                G.add_node(node_name, node_type="relationship")
        
        # Add edges
        edge_index = self.graph_data.edge_index.numpy()
        idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        
        for i in range(edge_index.shape[1]):
            src_idx, dst_idx = edge_index[:, i]
            src_node = idx_to_node[src_idx]
            dst_node = idx_to_node[dst_idx]
            G.add_edge(src_node, dst_node)
        
        return G

def main():
    st.set_page_config(page_title="Text2Cypher GNN-RAG Demo", page_icon="🧠")
    
    st.title("🧠 Text2Cypher GNN-RAG Demo")
    st.markdown("Convert natural language questions to Neo4j Cypher queries using Graph Neural Networks")
    
    # Initialize system
    @st.cache_resource
    def load_gnn_rag_system():
        return Text2CypherGNNRAG()
    
    try:
        rag_system = load_gnn_rag_system()
        
        # Sidebar
        with st.sidebar:
            st.header("🧠 GNN-RAG Architecture")
            st.markdown("""
            **Cách hoạt động:**
            1. Schema được chuyển thành graph
            2. Mỗi node type, property, relationship là một node
            3. GNN học embeddings từ cấu trúc graph
            4. Retrieval dựa trên similarity của embeddings
            """)
            
            st.header("📊 Database Schema")
            st.markdown("""
            **Nodes:** Product, Category, Supplier, Customer, Order
            
            **Relationships:**
            - Product → Category (PART_OF)
            - Supplier → Product (SUPPLIES)  
            - Customer → Order (PURCHASED)
            - Order → Product (ORDERS)
            """)
            
            # Visualization option
            if st.button("Visualize Schema Graph"):
                st.session_state.show_graph = True
            
            st.header("💡 Example Questions")
            for i, example in enumerate(rag_system.example_queries):
                if st.button(f"Example {i+1}", key=f"ex_{i}"):
                    st.session_state.question = example["question"]
        
        # Show graph visualization if requested
        if hasattr(st.session_state, 'show_graph') and st.session_state.show_graph:
            st.header("Schema Graph Visualization")
            try:
                G = rag_system.visualize_graph()
                
                # Create layout
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Draw different node types with different colors
                entity_nodes = [n for n in G.nodes() if n.startswith("NodeType:")]
                prop_nodes = [n for n in G.nodes() if n.startswith("Property:")]
                rel_nodes = [n for n in G.nodes() if n.startswith("Relationship:")]
                
                nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, 
                                     node_color='lightblue', node_size=1000, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=prop_nodes,
                                     node_color='lightgreen', node_size=500, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=rel_nodes,
                                     node_color='lightcoral', node_size=700, ax=ax)
                
                nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
                
                # Labels (simplified)
                labels = {n: n.split(':')[1][:10] + "..." if len(n.split(':')[1]) > 10 
                         else n.split(':')[1] for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
                
                ax.set_title("Database Schema as Graph")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error creating visualization: {e}")
        
        # Main interface
        st.header("Ask a Question")
        
        # Initialize session state
        if "question" not in st.session_state:
            st.session_state.question = ""
        
        question = st.text_area(
            "Enter your question in English:",
            value=st.session_state.question,
            height=100,
            placeholder="e.g., Show me all products with their categories"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            generate_btn = st.button("Generate Cypher", type="primary")
        
        with col2:
            clear_btn = st.button("Clear")
        
        if clear_btn:
            st.session_state.question = ""
            st.rerun()
        
        if generate_btn and question:
            with st.spinner("Generating Cypher query using GNN-RAG..."):
                try:
                    cypher_query, relevant_nodes = rag_system.generate_cypher(question)
                    
                    st.header("Generated Cypher Query")
                    st.code(cypher_query, language="cypher")
                    
                    # Show retrieved nodes
                    with st.expander("View Retrieved Graph Nodes"):
                        st.markdown("**Relevant nodes retrieved by GNN:**")
                        for node in relevant_nodes:
                            st.text(f"• {node}")
                        
                        context = rag_system.retriever.get_context_from_nodes(relevant_nodes)
                        st.markdown("**Generated Context:**")
                        st.text(context)
                    
                except Exception as e:
                    st.error(f"Error generating query: {str(e)}")
        
        # Information
        st.markdown("---")
        st.markdown("""
        **GNN-RAG Architecture:**
        1. Database schema được chuyển thành graph structure
        2. Mỗi node type, property và relationship thành graph nodes
        3. GNN (Graph Convolutional Network) học embeddings từ graph structure
        4. Query được embedded và so sánh với node embeddings
        5. Retrieved nodes được dùng làm context cho LLM
        
        **Advantages của GNN-RAG:**
        - Capture được structural relationships trong schema
        - Node embeddings chứa thông tin về neighbors
        - More context-aware retrieval
        """)
        
    except Exception as e:
        st.error(f"Failed to initialize GNN-RAG system: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        1. Install dependencies: `pip install torch torch-geometric networkx`
        2. Ensure Ollama is running: `ollama serve`
        3. Pull model: `ollama pull llama3.2`
        """)

if __name__ == "__main__":
    main() 