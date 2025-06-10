# Advanced GNN-RAG Setup

## 🚀 Cài đặt Dependencies

### Cách 1: Cài đặt từ requirements.txt
```bash
pip install -r requirements.txt
```

### Cách 2: Cài đặt từng package
```bash
# Core dependencies
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install sentence-transformers>=2.2.0
pip install scikit-learn>=1.3.0

# Existing dependencies
pip install networkx numpy dataclasses-json
```

## 🎯 Tính năng Advanced

### 1. **Sentence Transformers**
- Model: `all-MiniLM-L6-v2` 
- Embedding dimension: 384
- Encode thuộc tính văn bản của nodes

### 2. **PyTorch Geometric GCN**
- 2-layer Graph Convolutional Network
- Hidden dim: 128, Output dim: 64
- Self-supervised training với reconstruction loss

### 3. **Enhanced Path Finding**
- Sử dụng embedding similarity để tìm intermediate nodes
- Cosine similarity để đánh giá coherence
- Kết hợp graph structure và embedding space

### 4. **Smart Context Generation**
- Hiển thị coherence score cho mỗi path
- Tự động phân tích thông tin relevant với câu hỏi
- Enhanced formatting với emoji

## 🔄 Fallback Mode

Nếu không có advanced dependencies:
- Sẽ tự động fallback về embedding ngẫu nhiên
- Vẫn hoạt động đầy đủ các tính năng cơ bản
- Hiển thị warning message

## 📊 So sánh Performance

| Feature | Basic Mode | Advanced Mode |
|---------|------------|---------------|
| Embeddings | Random (64d) | Sentence-BERT (384d) + GCN (64d) |
| Path Finding | Graph structure only | Structure + Similarity |
| Context Quality | Basic | Enhanced with coherence |
| Training | None | GCN self-supervised |

## 🧪 Test Advanced Features

```bash
cd GNN-RAG
python demo_text2cypher_gnn_rag.py
```

Sẽ thấy:
- ✅ Advanced dependencies available
- 🔄 Loading sentence transformer
- 🔄 Training GCN...
- 📊 Embedding Info hiển thị dimensions thật

## 🚨 Troubleshooting

### Lỗi PyTorch Geometric
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Lỗi CUDA (nếu có GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Giảm hidden_dim từ 128 xuống 64
- Giảm epochs từ 50 xuống 20 