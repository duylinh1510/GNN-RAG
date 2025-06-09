# 🧠 Text2Cypher GNN-RAG Demo

Một demo đơn giản về việc sử dụng **Graph Neural Networks (GNN)** trong **Retrieval-Augmented Generation (RAG)** để chuyển đổi câu hỏi tự nhiên thành Cypher queries.

## 🎯 Mục tiêu

Thay vì sử dụng text-based RAG truyền thống (FAISS + text embeddings), project này sử dụng:
- **Graph structure** để represent database schema
- **Graph Neural Networks** để học structural embeddings
- **Graph-based retrieval** thay vì similarity search trên text chunks

## 🏗️ Architecture

### 1. Graph Construction
```
Database Schema → Graph Structure
├── Node Types: Product, Category, Supplier, Customer, Order
├── Properties: Mỗi property thành một graph node
├── Relationships: PART_OF, SUPPLIES, PURCHASED, ORDERS
└── Edges: Connect nodes với properties và relationships
```

### 2. GNN Model
```
Input: Initial node embeddings (từ sentence transformer)
      ↓
GNN Layers: Graph Convolutional Networks (GCN)
      ↓
Output: Learned embeddings chứa thông tin structural
```

### 3. Retrieval Process
```
User Question → Embedding → Similarity với GNN embeddings → Top-k nodes
```

### 4. Generation
```
Retrieved nodes → Context → LLM (Ollama) → Cypher Query
```

## 🚀 Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install PyTorch Geometric:**
```bash
# For CPU
pip install torch torch-geometric

# For GPU (nếu có CUDA)
pip install torch torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

3. **Setup Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Start server
ollama serve
```

## 💻 Usage

### Streamlit Web Demo
```bash
streamlit run gnn_rag_demo.py
```

### Console Demo
```bash
python gnn_console_demo.py
```

## 🔬 GNN-RAG vs Traditional RAG

| Aspect | Traditional RAG | GNN-RAG |
|--------|----------------|---------|
| **Data Representation** | Text chunks | Graph structure |
| **Embeddings** | Text embeddings | Structural embeddings |
| **Retrieval** | Similarity search | Graph-based retrieval |
| **Context Awareness** | Limited | Rich structural context |
| **Relationships** | Implicit | Explicit |

## 📊 Graph Schema

### Nodes
- **NodeType nodes**: Product, Category, Supplier, Customer, Order
- **Property nodes**: Mỗi property của mỗi node type
- **Relationship nodes**: PART_OF, SUPPLIES, PURCHASED, ORDERS

### Edges
- NodeType ↔ Properties (bidirectional)
- NodeType ↔ Relationships (through relationship nodes)

### Example Graph
```
Product ↔ productName
Product ↔ unitPrice
Product ↔ categoryID
Product ↔ [PART_OF] ↔ Category
Category ↔ categoryName
...
```

## 🎯 Advantages của GNN-RAG

1. **Structural Awareness**: GNN học được relationships giữa các entities
2. **Neighbor Information**: Node embeddings chứa thông tin từ neighbors
3. **Schema Understanding**: Model hiểu được cấu trúc database tốt hơn
4. **Context-Rich Retrieval**: Retrieved context meaningful hơn

## 🔧 Implementation Details

### GNN Model
- **Architecture**: Graph Convolutional Network (GCN)
- **Layers**: 2 layers (có thể customize)
- **Training**: Unsupervised - maximize similarity của connected nodes
- **Loss Function**: Negative cosine similarity

### Training Strategy
```python
# Simple reconstruction loss
edge_embeddings = model(graph)
src_emb = edge_embeddings[edge_index[0]]
dst_emb = edge_embeddings[edge_index[1]]
loss = -cosine_similarity(src_emb, dst_emb).mean()
```

### Retrieval Process
1. Encode user question với sentence transformer
2. Compute cosine similarity với tất cả node embeddings
3. Select top-k relevant nodes
4. Generate context từ selected nodes

## 📝 Example Queries

1. **"Show me all products with their categories"**
   - Retrieved nodes: NodeType:Product, Property:Product.productName, Relationship:Product-PART_OF->Category
   - Generated: `MATCH (p:Product)-[:PART_OF]->(c:Category) RETURN p.productName, c.categoryName`

2. **"Find suppliers from UK"**
   - Retrieved nodes: NodeType:Supplier, Property:Supplier.country, Property:Supplier.companyName
   - Generated: `MATCH (s:Supplier) WHERE s.country = 'UK' RETURN s.companyName, s.city`

## 🎨 Visualization

Demo bao gồm graph visualization để hiểu structure:
- **Blue nodes**: Entity types (Product, Category, etc.)
- **Green nodes**: Properties
- **Red nodes**: Relationships

## ⚡ Performance Notes

- **Graph size**: ~69 nodes (5 node types + properties + relationships)
- **Training time**: ~50 epochs, very fast
- **Inference**: Real-time
- **Memory**: Lightweight, chạy được trên CPU

## 🔮 Future Enhancements

1. **More sophisticated GNN architectures** (GraphSAGE, GAT)
2. **Multi-hop reasoning** trong graph
3. **Dynamic graph updates** khi schema thay đổi
4. **Query optimization** based on graph structure
5. **Support multiple databases** với different schemas

## 🐛 Troubleshooting

1. **PyTorch Geometric installation issues:**
   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
   ```

2. **Ollama connection issues:**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

3. **Memory issues:**
   - Giảm batch size trong training
   - Sử dụng smaller hidden dimensions

## 📚 References

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- [Neo4j Cypher](https://neo4j.com/docs/cypher-manual/)
- [Ollama](https://ollama.ai/)

---

Đây là một **proof-of-concept** đơn giản để demonstrate khả năng của GNN trong RAG systems. Có thể mở rộng thêm nhiều features và optimizations khác! 🚀 