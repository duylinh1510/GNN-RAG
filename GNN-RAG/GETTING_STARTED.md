# Hướng dẫn bắt đầu - Text2Cypher GNN-RAG

## 🚀 Cài đặt nhanh

### Bước 1: Cài đặt dependencies
```bash
pip install networkx numpy
```

### Bước 2: Chạy demo
```bash
cd GNN-RAG
python demo_text2cypher_gnn_rag.py
```

## 🎯 Text2Cypher GNN-RAG là gì?

**Text2Cypher GNN-RAG** là một hệ thống chuyển đổi câu hỏi tự nhiên thành Cypher query (ngôn ngữ truy vấn Neo4j) bằng cách sử dụng:

- **Text2Cypher**: Chuyển đổi từ ngôn ngữ tự nhiên sang Cypher
- **GNN (Graph Neural Network)**: Mạng neural graph để hiểu cấu trúc graph
- **RAG (Retrieval-Augmented Generation)**: Tìm kiếm thông tin từ graph làm context

## 🔄 Luồng hoạt động (5 bước)

```
Câu hỏi → Nhận diện thực thể → Tìm nodes → GNN tìm đường đi → Tạo context → Sinh Cypher
```

### Ví dụ cụ thể:

**Input**: "Tìm sản phẩm Chai"

1. **Nhận diện**: "sản phẩm" (Product), "Chai" (Product)
2. **Tìm node**: Product_P1 (Chai)
3. **GNN**: Tìm đường đi liên quan đến Chai
4. **Context**: Thông tin về Chai và các liên kết
5. **Output**: 
   ```cypher
   MATCH (p:Product)
   WHERE p.productName CONTAINS 'Chai'
   RETURN p.productName, p.unitPrice
   ```

## 📊 Dữ liệu mẫu

Demo sử dụng dữ liệu về **hệ thống quản lý đơn hàng**:

- 🛍️ **Products**: Chai, Chang, Aniseed Syrup
- 📁 **Categories**: Beverages, Condiments  
- 👥 **Customers**: Alfreds Futterkiste, Ana Trujillo
- 🏭 **Suppliers**: Exotic Liquids, New Orleans Cajun
- 📋 **Orders**: Đơn hàng của khách hàng

## 🎮 Cách sử dụng

### Menu chính:
1. **Câu hỏi mẫu** - Chọn từ 4 câu hỏi có sẵn
2. **Câu hỏi tùy chỉnh** - Nhập câu hỏi của bạn
3. **Xem cấu trúc graph** - Hiển thị thống kê
4. **Thoát**

### Câu hỏi mẫu bạn có thể thử:
- "Tìm thông tin về sản phẩm Chai"
- "Khách hàng nào đã mua đồ uống?"
- "Sản phẩm nào thuộc danh mục Beverages?"
- "Đơn hàng của khách hàng Alfreds"

## 🧠 Tại sao cần GNN?

**Vấn đề**: Làm sao tìm được thông tin liên quan trong graph phức tạp?

**Giải pháp GNN**:
- Hiểu cấu trúc graph và mối quan hệ
- Tìm đường đi tối ưu giữa các thực thể
- Cung cấp context phong phú cho việc sinh Cypher

## 🔍 Ví dụ chi tiết

```
Câu hỏi: "Khách hàng nào đã mua đồ uống?"

Bước 1: Nhận diện
- "khách hàng" → Customer
- "đồ uống" → Product (Beverages)

Bước 2: Tìm nodes
- Customer nodes: Customer_CUST1, Customer_CUST2
- Beverage products: Product_P1 (Chai), Product_P2 (Chang)

Bước 3: GNN tìm đường đi
- Customer_CUST1 → Order_O1 → Product_P1 (Chai)
- Customer_CUST2 → Order_O2 → Product_P2 (Chang)

Bước 4: Context
"Khách hàng Alfreds đã mua Chai, Ana Trujillo đã mua Chang..."

Bước 5: Cypher
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)
WHERE p.categoryName = 'Beverages'
RETURN cust.companyName, p.productName
```

## 🎯 Điểm mạnh của demo này

- ✅ **Đơn giản**: Không cần GPU hay LLM phức tạp
- ✅ **Rõ ràng**: Hiển thị từng bước xử lý  
- ✅ **Tương tác**: Menu console thân thiện
- ✅ **Thực tế**: Dữ liệu có ý nghĩa
- ✅ **Học tập**: Dễ hiểu cơ chế hoạt động

## 🚀 Bắt đầu ngay!

```bash
python demo_text2cypher_gnn_rag.py
```

Chọn option 1 và thử câu hỏi đầu tiên để xem demo hoạt động! 