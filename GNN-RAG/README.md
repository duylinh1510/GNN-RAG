# Advanced Text2Cypher GNN-RAG Demo

## 🎯 Mục đích
Demo này minh họa cách hoạt động của **Text2Cypher GNN-RAG** sử dụng **Sentence Transformers** và **PyTorch Geometric GCN** để chuyển đổi câu hỏi tự nhiên thành Cypher queries.

## 🧠 Công nghệ sử dụng

- **Sentence Transformers**: Encode text attributes thành embeddings 384D
- **PyTorch Geometric GCN**: Học enhanced node embeddings 64D  
- **NetworkX**: Quản lý graph structure
- **Cosine Similarity**: Đánh giá embedding similarity

## 📋 Luồng xử lý Advanced

### 1. **Entity Recognition (NER)**
```
"Khách hàng nào đã mua đồ uống?" 
↓
Entities: [khách hàng:Customer, đồ uống:Product, mua:Order]
```

### 2. **Node Mapping trong Knowledge Graph**
```
khách hàng → [Customer_CUST1, Customer_CUST2]
đồ uống → [Product_P1:Chai, Product_P2:Chang, Category_C1:Beverages]
```

### 3. **Advanced Embedding Generation**
```
Text Attributes → Sentence-BERT (384D) → GCN Training → Enhanced Embeddings (64D)

Ví dụ node text:
"company: Alfreds Futterkiste | contact: Maria Anders | city: Berlin | type: Customer"
↓ Sentence-BERT ↓
[0.12, -0.34, 0.67, ..., 0.89] (384 dimensions)
↓ GCN Training ↓  
[0.45, 0.23, -0.12, ..., 0.78] (64 dimensions)
```

### 4. **Intelligent Path Finding**
- **Method 1**: Graph shortest paths
- **Method 2**: Embedding similarity để tìm intermediate nodes
- **Scoring**: Kết hợp path length + embedding coherence + diversity

```
Embedding Similarity Example:
Customer_CUST1 ↔ Order_O1: similarity = 0.834
Order_O1 ↔ Product_P1: similarity = 0.723
Product_P1 ↔ Category_C1: similarity = 0.891
→ Path coherence = 0.816
```

### 5. **Enhanced Context Generation**
```
=== ENHANCED CONTEXT TỪ GNN-RAG ===

🎯 Thông tin liên quan đến câu hỏi:
  • Customer: Alfreds Futterkiste
  • Product: Chai
  • Category: Beverages

📍 Đường đi được tìm thấy:

Đường đi 1 (coherence: 0.816):
  Customer: Alfreds Futterkiste --[PURCHASED]--> Order: Order_O1 --[ORDERS]--> Product: Chai --[PART_OF]--> Category: Beverages
```

### 6. **Smart Cypher Generation**
```
Context + Question Analysis → Template Selection → Cypher Query

Output:
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)-[:PART_OF]->(c:Category)
WHERE c.categoryName = 'Beverages'
RETURN DISTINCT cust.contactName as CustomerName, 
       cust.companyName as Company,
       cust.city as City
ORDER BY cust.contactName
```

## 🏗️ Cấu trúc dữ liệu

Demo sử dụng schema từ file `schema.txt` với các thực thể:

- **Product**: Sản phẩm (tên, giá, category)
- **Category**: Danh mục (tên, mô tả)
- **Customer**: Khách hàng (tên, địa chỉ)
- **Supplier**: Nhà cung cấp (tên, liên hệ)
- **Order**: Đơn hàng (ngày, khách hàng)

### Relationships:
- `Product -[:PART_OF]-> Category`
- `Supplier -[:SUPPLIES]-> Product`
- `Customer -[:PURCHASED]-> Order`
- `Order -[:ORDERS]-> Product`

## 🚀 Cách sử dụng

### ⚙️ Hai chế độ hoạt động

#### 1. **Chế độ Advanced** (Recommended)
Sử dụng Sentence Transformers + PyTorch Geometric GCN
```bash
# Cài đặt advanced dependencies
pip install torch>=2.0.0 torch-geometric>=2.3.0 sentence-transformers>=2.2.0 scikit-learn>=1.3.0

# Cài đặt basic dependencies
pip install -r requirements.txt

# Chạy demo
cd GNN-RAG
python demo_text2cypher_gnn_rag.py
```

#### 2. **Chế độ Fallback** (Tự động)
Sử dụng embeddings đơn giản khi không có advanced libraries
```bash
# Chỉ cài đặt basic dependencies  
pip install -r requirements.txt

# Chạy demo (tự động fallback)
cd GNN-RAG
python demo_text2cypher_gnn_rag.py
```

### 🎮 Menu tương tác
1. **Chọn câu hỏi mẫu**: Test với câu hỏi được tối ưu
2. **Nhập câu hỏi tùy chỉnh**: Thử nghiệm tự do
3. **Xem cấu trúc knowledge graph**: Hiểu về dữ liệu và relationships
4. **Thoát**: Kết thúc demo

### 📊 Thông tin hiển thị
- **Entity extraction**: Các thực thể được nhận diện
- **Node matching**: Mapping entities → graph nodes  
- **Path finding**: Đường đi với điểm coherence
- **Enhanced context**: Context được tối ưu bởi GNN
- **Cypher query**: Query cuối cùng để truy vấn Neo4j
- **Embedding info**: Chi tiết về embeddings và GCN training

## 🧪 Ví dụ Demo Run

### Input
```
🔍 Câu hỏi: Khách hàng nào đã mua đồ uống?
```

### Output Flow
```
Bước 1: Xác định đối tượng trong câu hỏi...
Tìm thấy 3 đối tượng:
  - khách hàng (Customer) - độ tin cậy: 0.80
  - đồ uống (Product) - độ tin cậy: 0.80  
  - mua (Order) - độ tin cậy: 0.60

Bước 2: Tìm nodes trong knowledge graph...
Tìm thấy 5 nodes:
  - Customer_CUST1: Alfreds Futterkiste (Customer)
  - Customer_CUST2: Ana Trujillo (Customer)
  - Product_P1: Chai (Product)
  - Product_P2: Chang (Product)
  - Category_C1: Beverages (Category)

Bước 3: Tìm đường đi bằng Advanced GNN...
✅ Loaded sentence transformer với embedding dim: 384
🔄 Tạo text embeddings cho nodes...
🔄 Chuẩn bị dữ liệu PyTorch Geometric...
🔄 Khởi tạo và huấn luyện GCN...
  Epoch 0: Loss = 0.8234
  Epoch 10: Loss = 0.3456
  ...
✅ GCN training hoàn thành!

Tìm thấy 3 đường đi:
  - Điểm: 0.845 | Customer_CUST1 → Order_O1 → Product_P1 → Category_C1
  - Điểm: 0.723 | Customer_CUST2 → Order_O2 → Product_P2  
  - Điểm: 0.612 | Customer_CUST1 → Order_O1

Bước 4: Tạo enhanced context từ đường đi...
=== ENHANCED CONTEXT TỪ GNN-RAG ===

🎯 Thông tin liên quan đến câu hỏi:
  • Customer: Alfreds Futterkiste
  • Customer: Ana Trujillo  
  • Product: Chai
  • Category: Beverages

📍 Đường đi được tìm thấy:

Đường đi 1 (coherence: 0.816):
Customer: Alfreds Futterkiste --[PURCHASED]--> Order: Order_O1 --[ORDERS]--> Product: Chai --[PART_OF]--> Category: Beverages

Bước 5: Sinh Cypher query...
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)-[:PART_OF]->(c:Category)
WHERE c.categoryName = 'Beverages'
RETURN DISTINCT cust.contactName as CustomerName, 
       cust.companyName as Company,
       cust.city as City
ORDER BY cust.contactName

📊 Embedding Info:
  Text embedding dim: 384
  Enhanced embedding dim: 64
  GCN trained: True
```

## 🧪 Các câu hỏi test

### Tiếng Việt
- "Tìm thông tin về sản phẩm Chai"
- "Khách hàng nào đã mua đồ uống?"
- "Sản phẩm nào thuộc danh mục Beverages?"
- "Đơn hàng của khách hàng Alfreds"
- "List all products"
- "Liệt kê tất cả khách hàng"

### Tiếng Anh
- "Find products in category Beverages"
- "List all customers"
- "Show orders for customer Ana"
- "What products are available?"
- "Which customers bought drinks?"

## 🔧 Advanced Architecture

### Core Components

1. **EntityExtractor**: NLP-based entity recognition với keyword matching
2. **GraphBuilder**: Knowledge graph construction từ schema.txt
3. **AdvancedGCN**: PyTorch Geometric GCN (2 layers, 128→64 dims)
4. **SentenceEmbeddingExtractor**: Sentence-BERT text encoding 
5. **GNNPathFinder**: Advanced path finding với embedding similarity
6. **ContextGenerator**: Enhanced context generation với coherence scoring
7. **Text2CypherGenerator**: Template-based Cypher generation

### Embedding Pipeline
```
Node Attributes → Text Encoding → Sentence-BERT (384D) → GCN Training → Enhanced Embeddings (64D)
```

### Path Finding Algorithm
1. **Graph Structure Paths**: NetworkX shortest path
2. **Embedding Similarity**: Cosine similarity để tìm intermediate nodes
3. **Coherence Scoring**: Average pairwise similarity trong path
4. **Multi-criteria Ranking**: Path length + coherence + diversity

## 📚 Key Features

- **🧠 AI-Powered**: Sentence Transformers + Graph Neural Networks
- **🔄 Fallback Support**: Tự động chuyển về mode đơn giản nếu thiếu dependencies
- **🌐 Multilingual**: Hỗ trợ tiếng Việt và tiếng Anh  
- **📊 Rich Analytics**: Embedding dimensions, coherence scores, training metrics
- **🎯 Interactive**: Menu-driven với detailed step-by-step output
- **⚡ Real-time**: GCN training và inference trong demo

## ⚙️ System Requirements

### Minimum (Fallback mode)
- Python 3.8+
- NetworkX, NumPy

### Recommended (Advanced mode)  
- Python 3.8+
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- Sentence Transformers >= 2.2.0
- Scikit-learn >= 1.3.0

## 🔮 Possible Extensions

### Technical Improvements
- **Multi-hop reasoning**: Deeper GNN architectures
- **Attention mechanisms**: Graph Attention Networks (GAT)
- **Dynamic graphs**: Real-time graph updates
- **LLM integration**: Fine-tuned language models cho Cypher generation

### Data & Domain
- **Larger knowledge graphs**: Multi-domain schemas
- **Real Neo4j integration**: Direct database connections
- **Streaming data**: Real-time entity extraction
- **Multi-modal**: Image/document entity extraction

### Advanced Features  
- **Query optimization**: Cypher query planning
- **Explanation generation**: Natural language explanations
- **Interactive graph visualization**: Web-based UI
- **Benchmark evaluation**: Standard dataset testing 