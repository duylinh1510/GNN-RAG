# Text2Cypher RAG Demo

Một demo RAG (Retrieval-Augmented Generation) đơn giản để chuyển đổi câu hỏi tiếng Anh thành Cypher queries cho Neo4j, sử dụng các mô hình miễn phí.

## Tính năng

- 🔍 Chuyển đổi câu hỏi tự nhiên thành Cypher queries
- 🤖 Sử dụng Ollama với mô hình llama3.2 (miễn phí)
- 📚 RAG với vector similarity search
- 🎯 Hiểu schema Neo4j từ file schema.txt
- 🖥️ Giao diện Streamlit và console
- 📝 Các câu hỏi mẫu để test

## Cài đặt

### 1. Cài đặt Ollama

```bash
# Trên Windows (sử dụng winget)
winget install Ollama.Ollama

# Hoặc tải từ https://ollama.ai/download
```

### 2. Cài đặt mô hình llama3.2

```bash
ollama pull llama3.2
```

### 3. Khởi động Ollama

```bash
ollama serve
```

### 4. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng

### Giao diện Streamlit (Khuyến nghị)

```bash
streamlit run text2cypher_demo.py
```

Mở trình duyệt tại http://localhost:8501

### Giao diện Console

```bash
python console_demo.py
```

## Ví dụ câu hỏi

1. "Show me all products with their categories"
2. "Find suppliers from UK"
3. "Get orders with customer information"
4. "Show products with low stock"
5. "Find expensive products over 50"

## Cách hoạt động

1. **Embedding**: Câu hỏi được embed bằng sentence-transformers
2. **Retrieval**: Tìm kiếm thông tin schema liên quan bằng similarity search
3. **Generation**: Ollama (llama3.2) sinh Cypher query dựa trên context
4. **RAG Pipeline**: Kết hợp retrieval và generation để có kết quả chính xác

## Cấu trúc Database Schema

### Nodes
- **Product**: Thông tin sản phẩm
- **Category**: Danh mục sản phẩm  
- **Supplier**: Nhà cung cấp
- **Customer**: Khách hàng
- **Order**: Đơn hàng

### Relationships
- `(:Product)-[:PART_OF]->(:Category)`
- `(:Supplier)-[:SUPPLIES]->(:Product)`
- `(:Customer)-[:PURCHASED]->(:Order)`
- `(:Order)-[:ORDERS]->(:Product)`

## Troubleshooting

### Lỗi kết nối Ollama
```bash
# Kiểm tra Ollama đang chạy
ollama list

# Khởi động lại Ollama
ollama serve
```

### Lỗi mô hình không tìm thấy
```bash
# Cài đặt lại mô hình
ollama pull llama3.2
```

### Lỗi dependencies
```bash
# Cài đặt lại dependencies
pip install -r requirements.txt --upgrade
```

## Tùy chỉnh

### Thay đổi mô hình Ollama
Chỉnh sửa trong `text2cypher_demo.py`:
```python
self.llm = OllamaLLM(model="llama3.1")  # hoặc mô hình khác
```

### Thêm ví dụ mới
Thêm vào `example_queries` trong `text2cypher_demo.py`:
```python
{
    "question": "Your question here",
    "cypher": "MATCH (n) RETURN n LIMIT 10"
}
```

## Yêu cầu hệ thống

- Python 3.8+
- RAM: 4GB+ (để chạy llama3.2)
- Ollama đã cài đặt và chạy
- Kết nối internet (lần đầu tải mô hình)

## Giới hạn

- Chỉ hỗ trợ schema đã định nghĩa
- Độ chính xác phụ thuộc vào chất lượng mô hình
- Cần Ollama chạy local
- Chỉ xử lý câu hỏi tiếng Anh

## Mở rộng

- Thêm kết nối thực tế với Neo4j database
- Hỗ trợ nhiều ngôn ngữ
- Cải thiện prompt engineering
- Thêm validation cho Cypher queries
- Tích hợp với GraphRAG patterns 