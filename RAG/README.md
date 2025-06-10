# Giải thích Source Code - Hệ thống RAG Text2Cypher

Tài liệu này giải thích chi tiết cách hoạt động của source code trong hệ thống RAG Text2Cypher, từ kiến trúc tổng thể đến từng method cụ thể.

## 📁 Cấu trúc File

```
RAG/
├── text2cypher_demo.py    # File chính chứa toàn bộ logic
├── schema.txt            # Định nghĩa database schema
├── requirements.txt      # Python dependencies
└── README.md            # Tài liệu này
```

## 🏗️ Kiến trúc Source Code

### Class `Text2CypherRAG`

Đây là class chính chứa toàn bộ logic của hệ thống RAG:

```python
class Text2CypherRAG:
    def __init__(self):           # Khởi tạo các components
    def _load_schema_knowledge()  # Tạo knowledge base
    def _create_vector_store()    # Tạo FAISS vector store  
    def _get_relevant_context()   # Truy xuất context
    def _validate_cypher_query()  # Validation layer
    def generate_cypher()         # Method chính sinh Cypher
```
Cách Hệ Thống Hoạt Động
Hệ thống RAG Text2Cypher sử dụng phương pháp RAG (Retrieval-Augmented Generation) để:

Tìm thông tin liên quan: Dựa trên câu hỏi, hệ thống lấy thông tin từ cấu trúc cơ sở dữ liệu và các ví dụ mẫu.
Tạo câu truy vấn: Kết hợp thông tin tìm được với câu hỏi để tạo câu truy vấn Cypher chính xác.
Thành Phần Chính (Text2CypherRAG)
Lớp Text2CypherRAG chứa toàn bộ logic, với các chức năng:

Khởi tạo hệ thống
Tải thông tin cơ sở dữ liệu
Tạo kho lưu trữ thông tin
Tìm thông tin liên quan đến câu hỏi
Kiểm tra câu truy vấn
Sinh câu truy vấn Cypher
## 🔧 Chi tiết các Method

### 1. `__init__(self)` - Khởi tạo hệ thống

```python
def __init__(self):
    # Khởi tạo Ollama LLM
    self.llm = OllamaLLM(model="llama3.2")
    
    # Khởi tạo embedding model
    self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Định nghĩa example queries (dùng để training)
    self.example_queries = [...]
    
    # Tạo knowledge base và vector store
    self.schema_docs = self._load_schema_knowledge()
    self.vector_store = self._create_vector_store()
```

**Nhiệm vụ**: Khởi tạo tất cả components cần thiết
- **LLM**: Ollama với model llama3.2 để sinh Cypher
- **Embeddings**: HuggingFace model để chuyển text thành vectors
- **Example queries**: 5 cặp question-cypher để làm training data
- **Knowledge base**: Load schema và tạo vector store

### 2. `_load_schema_knowledge(self)` - Tạo Knowledge Base

```python
def _load_schema_knowledge(self) -> List[Document]:
    schema_docs = []
    
    # Đọc schema từ file
    with open('schema.txt', 'r', encoding='utf-8') as f:
        schema_content = f.read()
    
    # Tạo các documents cho:
    nodes_info = """Node Types and Properties: ..."""
    relationships_info = """Relationships: ..."""  
    cypher_patterns = """Common Cypher Query Patterns: ..."""
    
    # Combine thành documents
    schema_docs.extend([
        Document(page_content=schema_content, metadata={"type": "schema"}),
        Document(page_content=nodes_info, metadata={"type": "nodes"}),
        Document(page_content=relationships_info, metadata={"type": "relationships"}),
        Document(page_content=cypher_patterns, metadata={"type": "patterns"})
    ])
    
    # Thêm example queries
    for example in self.example_queries:
        doc_content = f"Question: {example['question']}\nCypher: {example['cypher']}"
        schema_docs.append(Document(page_content=doc_content, metadata={"type": "example"}))
    
    return schema_docs
```

**Nhiệm vụ**: Tạo ra knowledge base từ nhiều nguồn
- **schema.txt**: Raw schema definition  
- **nodes_info**: Thông tin chi tiết về các node types
- **relationships_info**: Mô tả các relationships
- **cypher_patterns**: Các pattern Cypher thường dùng
- **example_queries**: 5 cặp Q&A mẫu

**Output**: List các Document objects, mỗi document có content và metadata

### 3. `_create_vector_store(self)` - Tạo Vector Store

```python
def _create_vector_store(self):
    return FAISS.from_documents(self.schema_docs, self.embeddings)
```

**Nhiệm vụ**: Chuyển đổi documents thành vector store
- **Input**: `self.schema_docs` (list documents) + `self.embeddings` (embedding model)
- **Process**: Mỗi document được chuyển thành vector embedding
- **Output**: FAISS vector store có thể tìm kiếm similarity

### 4. `_get_relevant_context(self, question, k=3)` - Truy xuất Context

```python
def _get_relevant_context(self, question: str, k: int = 3) -> str:
    # Similarity search
    docs = self.vector_store.similarity_search(question, k=k)
    
    # Combine content
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
```

**Nhiệm vụ**: Tìm thông tin liên quan đến câu hỏi
- **Input**: Câu hỏi của user + số lượng documents cần lấy (k=3)
- **Process**: 
  1. Chuyển question thành vector
  2. Tìm k documents có cosine similarity cao nhất
  3. Ghép nội dung các documents lại
- **Output**: String chứa context liên quan

### 5. `_validate_cypher_query(self, cypher_query)` - Validation Layer

```python
def _validate_cypher_query(self, cypher_query: str) -> str:
    # Định nghĩa valid entities
    valid_nodes = {'Product', 'Category', 'Supplier', 'Customer', 'Order'}
    valid_relationships = {'PART_OF', 'SUPPLIES', 'PURCHASED', 'ORDERS'}
    
    # Regex patterns để extract entities
    node_pattern = r'\(\w*:([A-Z][A-Z_a-z]*)\)'
    rel_pattern = r'\[:([A-Z_]+)\]'
    
    # Kiểm tra nodes
    found_nodes = re.findall(node_pattern, cypher_query.upper())
    for node in found_nodes:
        if node not in [n.upper() for n in valid_nodes]:
            return f"ERROR: Node '{node}' không tồn tại..."
    
    # Kiểm tra relationships
    found_rels = re.findall(rel_pattern, cypher_query.upper()) 
    for rel in found_rels:
        if rel not in [r.upper() for r in valid_relationships]:
            return f"ERROR: Relationship '{rel}' không tồn tại..."
            
    return cypher_query
```

**Nhiệm vụ**: Kiểm tra Cypher query có hợp lệ với schema không
- **Input**: Cypher query string từ LLM
- **Process**:
  1. Extract tất cả node labels bằng regex
  2. Extract tất cả relationships bằng regex  
  3. So sánh với danh sách valid entities
  4. Trả về error nếu tìm thấy invalid entities
- **Output**: Cypher query gốc hoặc error message

### 6. `generate_cypher(self, question)` - Method chính

```python
def generate_cypher(self, question: str) -> str:
    # 1. Lấy context liên quan
    context = self._get_relevant_context(question)
    
    # 2. Tạo prompt template với rules nghiêm ngặt
    prompt_template = PromptTemplate(...)
    
    # 3. Sinh query bằng LLM
    prompt = prompt_template.format(context=context, question=question)
    cypher_query = self.llm.invoke(prompt)
    
    # 4. Clean up response
    cypher_query = cypher_query.strip()
    if cypher_query.startswith("```"):
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    
    # 5. Kiểm tra nếu LLM đã trả về error
    if cypher_query.startswith("ERROR:"):
        return cypher_query
    
    # 6. Validation layer cuối cùng
    validated_query = self._validate_cypher_query(cypher_query)
    
    return validated_query
```

**Nhiệm vụ**: Orchestrate toàn bộ pipeline RAG
- **Step 1**: Retrieval - Lấy context từ vector store
- **Step 2**: Augmentation - Kết hợp context với question thành prompt
- **Step 3**: Generation - LLM sinh Cypher query
- **Step 4**: Validation - Kiểm tra query hợp lệ

## 🔄 Flow xử lý từ Input đến Output

```
User nhập: "Show products with low stock"
    ↓
📋 _get_relevant_context():
    ├─ Embed question thành vector
    ├─ Similarity search trong FAISS
    ├─ Lấy top-3 documents liên quan:
    │  ├─ Document về Product node properties
    │  ├─ Document về Cypher patterns
    │  └─ Example query có filtering
    └─ Combine thành context string
    ↓
🤖 generate_cypher():
    ├─ Tạo prompt với context + question + rules
    ├─ Gửi tới Ollama LLM
    ├─ Nhận response: "MATCH (p:Product) WHERE p.unitsInStock < 10 RETURN p.productName, p.unitsInStock"
    └─ Clean up formatting
    ↓
✅ _validate_cypher_query():
    ├─ Extract nodes: ["PRODUCT"] → Valid ✓
    ├─ Extract relationships: [] → None, OK ✓
    └─ Return query gốc
    ↓
📤 Output: "MATCH (p:Product) WHERE p.unitsInStock < 10 RETURN p.productName, p.unitsInStock"
```

## 🧠 Validation Layer - 2 lớp

### Layer 1: LLM Self-Validation
- **Prompt Engineering**: Rules nghiêm ngặt trong prompt
- **Entity Mapping**: Explicit mapping valid/invalid entities  
- **Error Templates**: Hướng dẫn LLM trả về error message

### Layer 2: Regex Validation  
- **Backup Protection**: Nếu LLM bypass rules
- **Pattern Matching**: Extract nodes và relationships
- **Schema Enforcement**: Chỉ cho phép entities có trong schema

## 🛡️ Error Handling Strategy

```python
# Case 1: LLM tự detect invalid entities
Input: "find employees"
LLM Output: "ERROR: Entities not found in schema. Available entities: Product, Category, Supplier, Customer, Order"

# Case 2: LLM bypass nhưng regex bắt được
Input: "show departments"  
LLM Output: "MATCH (d:Department) RETURN d"
Regex Validation: "ERROR: Node 'DEPARTMENT' không tồn tại trong schema..."

# Case 3: Valid query pass through
Input: "show products"
LLM Output: "MATCH (p:Product) RETURN p"
Validation: Pass ✓
```

## 🚀 Performance Considerations

### Memory Usage
- **FAISS Vector Store**: ~10MB (cho schema documents)
- **Ollama llama3.2**: ~2GB RAM
- **HuggingFace Embeddings**: ~100MB

### Response Time
- **Embedding**: ~50ms (cho 1 question)
- **Similarity Search**: ~5ms (trong FAISS)
- **LLM Generation**: ~2-5s (tùy hardware)
- **Validation**: ~1ms (regex)

### Optimization Tips
- Vector store được cache sau lần đầu load
- Embedding model cache trong memory
- Ollama server persistent connection

## 🔧 Customization Points

### Thay đổi Schema
```python
# File: schema.txt
# Update node types, relationships, properties
# Restart program để reload vector store
```

### Tune Retrieval
```python
# Method: _get_relevant_context()
def _get_relevant_context(self, question: str, k: int = 5):  # Tăng k
    docs = self.vector_store.similarity_search(question, k=k)
```

### Improve Validation
```python
# Method: _validate_cypher_query()
# Thêm property validation
# Thêm syntax validation
# Thêm business logic validation
```

### Enhanced Prompting
```python
# Method: generate_cypher()
# Cải thiện prompt template
# Thêm few-shot examples
# Thêm chain-of-thought reasoning
```

## 📊 Dependencies Explanation

### Core Libraries
```python
from langchain_ollama import OllamaLLM              # LLM interface
from langchain.prompts import PromptTemplate        # Prompt management
from langchain.schema import Document               # Document structure
from langchain_community.vectorstores import FAISS # Vector database
from langchain_community.embeddings import HuggingFaceEmbeddings # Text embeddings
```

### Why these choices?
- **Ollama**: Free, local LLM runtime
- **FAISS**: Fast similarity search, Facebook's library
- **HuggingFace**: Free embedding models
- **LangChain**: Unified interface cho RAG pipeline

## 🎯 Cách chạy và test

### Khởi chạy
```bash
cd RAG
python text2cypher_demo.py
```

### Test cases đề xuất
```
✅ Valid queries:
- "show all products"
- "find suppliers from UK" 
- "get customer orders"

❌ Invalid entities (should show ERROR):
- "find employees"
- "show users"
- "list departments"
```

---

**Lưu ý**: Source code được thiết kế modular, dễ mở rộng và maintain. Mỗi method có responsibility rõ ràng và có thể test độc lập. 