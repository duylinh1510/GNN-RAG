"""
Demo Text2Cypher GNN-RAG đơn giản
=====================================

Demo này minh họa luồng xử lý của Text2Cypher GNN-RAG:
1. Xác định đối tượng trong câu hỏi
2. Tìm node đối tượng trong graph
3. Sử dụng GNN để tìm đường đi giữa các node
4. Xuất đường đi làm context cho việc sinh Cypher query

Dữ liệu mẫu: Hệ thống quản lý đơn hàng (Products, Categories, Suppliers, Customers, Orders)
"""

import re
import json
from typing import List, Dict, Tuple, Set
import networkx as nx
import numpy as np
from dataclasses import dataclass

# Advanced imports (sẽ dùng khi có dependencies)
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_AVAILABLE = True
    print("✅ Advanced dependencies available (PyTorch, PyG, SentenceTransformers)")
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️ Advanced dependencies not found. Install: torch, torch-geometric, sentence-transformers, scikit-learn")

@dataclass
class Entity:
    """Lớp đại diện cho một thực thể được nhận diện"""
    name: str
    entity_type: str
    confidence: float

class EntityExtractor:
    """Bước 1: Xác định đối tượng trong câu hỏi"""
    
    def __init__(self):
        # Từ điển các từ khóa cho từng loại thực thể
        self.entity_keywords = {
            'Product': [
                'sản phẩm', 'hàng hóa', 'mặt hàng', 'chai', 'chang', 'beer', 'bia',
                'product', 'products', 'item', 'goods', 'đồ uống', 'thức ăn', 'food', 'drink'
            ],
            'Category': [
                'danh mục', 'loại', 'category', 'nhóm', 'phân loại',
                'beverages', 'đồ uống', 'condiments', 'gia vị', 'seafood', 'hải sản'
            ],
            'Customer': [
                'khách hàng', 'customer', 'customers','client', 'người mua', 'công ty'
            ],
            'Supplier': [
                'nhà cung cấp', 'supplier', 'vendor', 'nhà phân phối'
            ],
            'Order': [
                'đơn hàng', 'order', 'orders', 'hóa đơn', 'mua', 'bán', 'đặt hàng'
            ]
        }
    
    def extract_entities(self, question: str) -> List[Entity]:
        """Trích xuất các thực thể từ câu hỏi"""
        question_lower = question.lower()
        entities = []
        
        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    confidence = 0.8 if len(keyword) > 3 else 0.6
                    entities.append(Entity(keyword, entity_type, confidence))
        
        # Loại bỏ trùng lặp và sắp xếp theo độ tin cậy
        unique_entities = {}
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return sorted(unique_entities.values(), key=lambda x: x.confidence, reverse=True)

class GraphBuilder:
    """Bước 2: Xây dựng và quản lý knowledge graph"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_attributes = {}
        self.build_sample_graph()
    
    def build_sample_graph(self):
        """Xây dựng graph mẫu từ schema"""
        # Thêm nodes mẫu
        sample_data = {
            'Product': [
                {'id': 'P1', 'name': 'Chai', 'price': 18.0, 'categoryID': 'C1'},
                {'id': 'P2', 'name': 'Chang', 'price': 19.0, 'categoryID': 'C1'},
                {'id': 'P3', 'name': 'Aniseed Syrup', 'price': 10.0, 'categoryID': 'C2'}
            ],
            'Category': [
                {'id': 'C1', 'name': 'Beverages', 'description': 'Soft drinks, coffees, teas, beers, and ales'},
                {'id': 'C2', 'name': 'Condiments', 'description': 'Sweet and savory sauces'}
            ],
            'Customer': [
                {'id': 'CUST1', 'companyName': 'Alfreds Futterkiste', 'contactName': 'Maria Anders', 'city': 'Berlin', 'country': 'Germany', 'phone': '030-0074321'},
                {'id': 'CUST2', 'companyName': 'Ana Trujillo', 'contactName': 'Ana Trujillo', 'city': 'Mexico D.F.', 'country': 'Mexico', 'phone': '(5) 555-4729'}
            ],
            'Supplier': [
                {'id': 'S1', 'companyName': 'Exotic Liquids', 'contactName': 'Charlotte Cooper', 'city': 'London', 'country': 'UK', 'phone': '(171) 555-2222'},
                {'id': 'S2', 'companyName': 'New Orleans Cajun', 'contactName': 'Shelley Burke', 'city': 'New Orleans', 'country': 'USA', 'phone': '(100) 555-4822'}
            ],
            'Order': [
                {'id': 'O1', 'orderDate': '2024-01-15', 'customerID': 'CUST1'},
                {'id': 'O2', 'orderDate': '2024-01-16', 'customerID': 'CUST2'}
            ]
        }
        
        # Thêm nodes vào graph
        for node_type, nodes in sample_data.items():
            for node in nodes:
                node_id = f"{node_type}_{node['id']}"
                self.graph.add_node(node_id, type=node_type, **node)
                self.node_attributes[node_id] = {**node, 'type': node_type}
        
        # Thêm relationships
        relationships = [
            ('Product_P1', 'Category_C1', 'PART_OF'),
            ('Product_P2', 'Category_C1', 'PART_OF'),
            ('Product_P3', 'Category_C2', 'PART_OF'),
            ('Supplier_S1', 'Product_P1', 'SUPPLIES'),
            ('Supplier_S2', 'Product_P2', 'SUPPLIES'),
            ('Customer_CUST1', 'Order_O1', 'PURCHASED'),
            ('Customer_CUST2', 'Order_O2', 'PURCHASED'),
            ('Order_O1', 'Product_P1', 'ORDERS'),
            ('Order_O2', 'Product_P2', 'ORDERS')
        ]
        
        for source, target, rel_type in relationships:
            self.graph.add_edge(source, target, relation=rel_type)
    
    def find_entity_nodes(self, entities: List[Entity]) -> List[str]:
        """Tìm các node trong graph tương ứng với entities"""
        matched_nodes = []
        
        for entity in entities:
            for node_id, attrs in self.node_attributes.items():
                if attrs['type'] == entity.entity_type:
                    # Nếu là từ khóa chung (như "product", "customer"), lấy tất cả nodes cùng loại
                    if entity.name.lower() in ['product', 'sản phẩm', 'customer', 'khách hàng', 
                                              'category', 'danh mục', 'supplier', 'nhà cung cấp', 
                                              'order', 'đơn hàng']:
                        matched_nodes.append(node_id)
                    # Xử lý đặc biệt cho "đồ uống" - tìm Category Beverages và Product liên quan
                    elif entity.name.lower() in ['đồ uống', 'drink', 'beverage', 'beverages']:
                        if attrs['type'] == 'Category' and 'beverages' in str(attrs.get('name', '')).lower():
                            matched_nodes.append(node_id)
                        elif attrs['type'] == 'Product' and attrs.get('categoryID') == 'C1':  # C1 = Beverages
                            matched_nodes.append(node_id)
                    # Nếu là tên cụ thể, tìm chính xác
                    elif (entity.name.lower() in str(attrs.get('name', '')).lower() or
                          entity.name.lower() in str(attrs.get('companyName', '')).lower() or
                          entity.name.lower() in str(attrs.get('description', '')).lower()):
                        matched_nodes.append(node_id)
        
        # Loại bỏ trùng lặp
        return list(set(matched_nodes))

class AdvancedGCN(torch.nn.Module):
    """Graph Convolutional Network để học node embeddings"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2):
        super(AdvancedGCN, self).__init__()
        self.num_layers = num_layers
        
        # Các layers GCN
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        """Forward pass qua GCN"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class GNNPathFinder:
    """Bước 3: Sử dụng GNN với sentence-transformers và PyTorch Geometric"""
    
    def __init__(self, graph: nx.Graph, graph_builder):
        self.graph = graph
        self.graph_builder = graph_builder
        
        if not ADVANCED_AVAILABLE:
            print("⚠️ Fallback to simple embeddings - install advanced dependencies for full features")
            self.node_embeddings = self._initialize_simple_embeddings()
            self.enhanced_embeddings = self.node_embeddings
        else:
            # Tạo sentence embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ Loaded sentence transformer với embedding dim: {self.sentence_model.get_sentence_embedding_dimension()}")
            
            # Tạo text embeddings cho nodes
            self.text_embeddings = self._create_text_embeddings()
            
            # Chuẩn bị dữ liệu PyTorch Geometric
            self.pyg_data = self._prepare_pyg_data()
            
            # Train GCN và tạo enhanced embeddings
            self.gcn_model = self._train_gcn()
            self.enhanced_embeddings = self._create_enhanced_embeddings()
    
    def _initialize_simple_embeddings(self) -> Dict[str, np.ndarray]:
        """Fallback: Khởi tạo embeddings đơn giản cho các nodes"""
        embeddings = {}
        embedding_dim = 64
        
        for node in self.graph.nodes():
            embeddings[node] = np.random.normal(0, 1, embedding_dim)
        
        return embeddings
    
    def _encode_node_text(self, node_attrs: Dict) -> str:
        """Chuyển đổi thuộc tính node thành text để encode"""
        text_parts = []
        
        # Lấy các thuộc tính quan trọng
        if 'name' in node_attrs:
            text_parts.append(f"name: {node_attrs['name']}")
        if 'companyName' in node_attrs:
            text_parts.append(f"company: {node_attrs['companyName']}")
        if 'contactName' in node_attrs:
            text_parts.append(f"contact: {node_attrs['contactName']}")
        if 'description' in node_attrs:
            text_parts.append(f"description: {node_attrs['description']}")
        if 'city' in node_attrs:
            text_parts.append(f"city: {node_attrs['city']}")
        if 'country' in node_attrs:
            text_parts.append(f"country: {node_attrs['country']}")
        if 'type' in node_attrs:
            text_parts.append(f"type: {node_attrs['type']}")
        
        # Nếu không có text, dùng type
        if not text_parts and 'type' in node_attrs:
            text_parts.append(node_attrs['type'])
        
        return " | ".join(text_parts) if text_parts else "unknown node"
    
    def _create_text_embeddings(self) -> Dict[str, np.ndarray]:
        """Tạo text embeddings bằng sentence-transformers"""
        print("🔄 Tạo text embeddings cho nodes...")
        
        node_texts = []
        node_ids = []
        
        for node_id, attrs in self.graph_builder.node_attributes.items():
            text = self._encode_node_text(attrs)
            node_texts.append(text)
            node_ids.append(node_id)
            print(f"  📝 {node_id}: {text}")
        
        # Encode tất cả texts
        print("🔄 Encoding với sentence transformer...")
        embeddings_array = self.sentence_model.encode(node_texts, show_progress_bar=True)
        
        # Chuyển về dict
        embeddings_dict = {}
        for i, node_id in enumerate(node_ids):
            embeddings_dict[node_id] = embeddings_array[i]
        
        print(f"✅ Đã tạo text embeddings cho {len(embeddings_dict)} nodes")
        return embeddings_dict
    
    def _prepare_pyg_data(self) -> Data:
        """Chuẩn bị dữ liệu cho PyTorch Geometric"""
        print("🔄 Chuẩn bị dữ liệu PyTorch Geometric...")
        
        # Tạo mapping từ node_id sang index
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        self.node_to_idx = node_to_idx
        
        # Node features (text embeddings)
        node_features = []
        for node in self.graph.nodes():
            node_features.append(self.text_embeddings[node])
        
        x = torch.FloatTensor(np.array(node_features))
        
        # Edge index
        edge_index = []
        for edge in self.graph.edges():
            src_idx = node_to_idx[edge[0]]
            dst_idx = node_to_idx[edge[1]]
            edge_index.append([src_idx, dst_idx])
            edge_index.append([dst_idx, src_idx])  # Undirected graph
        
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        
        data = Data(x=x, edge_index=edge_index)
        print(f"✅ PyG Data: {data.num_nodes} nodes, {data.num_edges} edges")
        
        return data
    
    def _train_gcn(self) -> AdvancedGCN:
        """Huấn luyện GCN model"""
        print("🔄 Khởi tạo và huấn luyện GCN...")
        
        input_dim = self.sentence_model.get_sentence_embedding_dimension()
        model = AdvancedGCN(input_dim=input_dim, hidden_dim=128, output_dim=64)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(self.pyg_data.x, self.pyg_data.edge_index)
            
            # Self-supervised loss: reconstruct node similarities
            reconstructed = torch.mm(embeddings, embeddings.t())
            original_sim = torch.mm(self.pyg_data.x, self.pyg_data.x.t())
            
            loss = F.mse_loss(reconstructed, original_sim)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        print("✅ GCN training hoàn thành!")
        model.eval()
        return model
    
    def _create_enhanced_embeddings(self) -> Dict[str, np.ndarray]:
        """Tạo enhanced embeddings từ trained GCN"""
        print("🔄 Tạo enhanced embeddings từ GCN...")
        
        with torch.no_grad():
            enhanced_emb = self.gcn_model(self.pyg_data.x, self.pyg_data.edge_index)
        
        # Chuyển về dict
        enhanced_embeddings = {}
        for i, node in enumerate(self.graph.nodes()):
            enhanced_embeddings[node] = enhanced_emb[i].numpy()
        
        print("✅ Enhanced embeddings đã sẵn sàng!")
        return enhanced_embeddings
    
    def find_paths(self, source_nodes: List[str], max_hops: int = 4) -> List[List[str]]:
        """Tìm đường đi bằng embedding similarity và graph structure"""
        if len(source_nodes) < 2:
            return [source_nodes] if source_nodes else []
        
        print("🔄 Tìm đường đi bằng embedding similarity...")
        all_paths = []
        
        # Method 1: Tìm đường đi trực tiếp như trước
        for i in range(len(source_nodes)):
            for j in range(i + 1, len(source_nodes)):
                try:
                    path = nx.shortest_path(self.graph, source_nodes[i], source_nodes[j])
                    if len(path) <= max_hops + 1:
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Method 2: Tìm nodes trung gian dựa trên embedding similarity (nếu có advanced features)
        if ADVANCED_AVAILABLE and hasattr(self, 'enhanced_embeddings'):
            intermediate_nodes = self._find_relevant_intermediate_nodes(source_nodes)
            
            for intermediate in intermediate_nodes[:3]:  # Top 3 most relevant
                for source in source_nodes:
                    try:
                        if intermediate != source:
                            path = nx.shortest_path(self.graph, source, intermediate)
                            if len(path) <= max_hops and path not in all_paths:
                                all_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # Method 3: Fallback logic cho trường hợp không có advanced features
        else:
            node_types = set()
            for node in source_nodes:
                node_type = self.graph.nodes[node].get('type', '')
                node_types.add(node_type)
            
            # Nếu có Customer và Product/Category, tìm đường đi thông qua Order
            if 'Customer' in node_types and ('Product' in node_types or 'Category' in node_types):
                customers = [n for n in source_nodes if self.graph.nodes[n].get('type') == 'Customer']
                products = [n for n in source_nodes if self.graph.nodes[n].get('type') == 'Product']
                categories = [n for n in source_nodes if self.graph.nodes[n].get('type') == 'Category']
                
                for customer in customers:
                    for product in products + categories:
                        try:
                            path = nx.shortest_path(self.graph, customer, product)
                            if len(path) <= max_hops + 1:
                                all_paths.append(path)
                        except nx.NetworkXNoPath:
                            continue
        
        return all_paths
    
    def _find_relevant_intermediate_nodes(self, source_nodes: List[str], top_k: int = 5) -> List[str]:
        """Tìm nodes trung gian dựa trên embedding similarity"""
        if not source_nodes:
            return []
        
        # Tính embedding trung bình của source nodes
        source_embeddings = [self.enhanced_embeddings[node] for node in source_nodes]
        avg_source_embedding = np.mean(source_embeddings, axis=0).reshape(1, -1)
        
        # Tính similarity với tất cả nodes
        similarities = {}
        for node_id, embedding in self.enhanced_embeddings.items():
            if node_id not in source_nodes:
                sim = cosine_similarity(avg_source_embedding, embedding.reshape(1, -1))[0][0]
                similarities[node_id] = sim
        
        # Trả về top-k similar nodes
        sorted_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        print(f"  🎯 Top relevant intermediate nodes:")
        for node, sim in sorted_nodes[:top_k]:
            attrs = self.graph_builder.node_attributes[node]
            display_name = attrs.get('name') or attrs.get('companyName') or node
            print(f"    - {node} ({display_name}): similarity = {sim:.3f}")
        
        return [node for node, sim in sorted_nodes[:top_k]]
    
    def score_paths(self, paths: List[List[str]]) -> List[Tuple[List[str], float]]:
        """Đánh giá độ quan trọng của các đường đi bằng embedding similarity"""
        scored_paths = []
        
        for path in paths:
            if len(path) < 2:
                scored_paths.append((path, 0.0))
                continue
            
            # Base score (ngược với độ dài)
            base_score = 1.0 / len(path)
            
            # Embedding coherence score (nếu có advanced features)
            if ADVANCED_AVAILABLE and hasattr(self, 'enhanced_embeddings'):
                coherence_score = self._calculate_path_coherence(path)
            else:
                coherence_score = 0.0
            
            # Diversity bonus
            node_types = set()
            for node in path:
                node_type = self.graph.nodes[node].get('type', '')
                node_types.add(node_type)
            diversity_bonus = len(node_types) * 0.1
            
            final_score = base_score + coherence_score + diversity_bonus
            scored_paths.append((path, final_score))
        
        return sorted(scored_paths, key=lambda x: x[1], reverse=True)
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """Tính coherence của path dựa trên embedding similarity"""
        if len(path) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(path) - 1):
            emb1 = self.enhanced_embeddings[path[i]].reshape(1, -1)
            emb2 = self.enhanced_embeddings[path[i + 1]].reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(sim)
        
        return np.mean(similarities) * 0.5  # Weight factor

class ContextGenerator:
    """Bước 4: Sinh context từ đường đi với enhanced embeddings"""
    
    def __init__(self, graph: nx.Graph, embeddings: Dict[str, np.ndarray] = None):
        self.graph = graph
        self.embeddings = embeddings
    
    def generate_context(self, paths: List[List[str]], question: str = "") -> str:
        """Tạo enhanced context từ các đường đi và embeddings"""
        if not paths:
            return "Không tìm thấy đường đi liên quan trong knowledge graph."
        
        context_parts = []
        if ADVANCED_AVAILABLE and self.embeddings:
            context_parts.append("=== ENHANCED CONTEXT TỪ GNN-RAG ===\n")
            
            # Nếu có câu hỏi, tìm nodes most relevant
            if question:
                relevant_info = self._find_question_relevant_info(question, paths)
                context_parts.append("🎯 Thông tin liên quan đến câu hỏi:")
                context_parts.append(relevant_info)
                context_parts.append("")
        else:
            context_parts.append("=== THÔNG TIN TỪ KNOWLEDGE GRAPH ===\n")
        
        context_parts.append("📍 Đường đi được tìm thấy:")
        
        for i, path in enumerate(paths[:3]):  # Chỉ lấy 3 đường đi tốt nhất
            coherence_info = ""
            if ADVANCED_AVAILABLE and self.embeddings:
                coherence = self._calculate_path_coherence(path)
                coherence_info = f" (coherence: {coherence:.3f})"
            
            context_parts.append(f"\nĐường đi {i+1}{coherence_info}:")
            
            path_description = []
            for j, node in enumerate(path):
                node_attrs = self.graph.nodes[node]
                node_type = node_attrs.get('type', 'Unknown')
                node_name = node_attrs.get('name') or node_attrs.get('companyName') or node
                
                path_description.append(f"{node_type}: {node_name}")
                
                if j < len(path) - 1:
                    # Thêm thông tin về relationship
                    if self.graph.has_edge(path[j], path[j+1]):
                        edge_data = self.graph.edges[path[j], path[j+1]]
                        relation = edge_data.get('relation', 'CONNECTED_TO')
                        path_description.append(f" --[{relation}]--> ")
            
            context_parts.append("  " + " ".join(path_description))
        
        return "\n".join(context_parts)
    
    def _find_question_relevant_info(self, question: str, paths: List[List[str]]) -> str:
        """Tìm thông tin relevant với câu hỏi"""
        question_lower = question.lower()
        relevant_info = []
        
        # Phân tích các nodes trong paths
        all_nodes = set()
        for path in paths:
            all_nodes.update(path)
        
        for node in all_nodes:
            attrs = self.graph.nodes[node]
            node_type = attrs.get('type', '')
            
            if 'customer' in question_lower and node_type == 'Customer':
                name = attrs.get('companyName') or attrs.get('contactName')
                relevant_info.append(f"  • Customer: {name}")
            elif 'product' in question_lower and node_type == 'Product':
                name = attrs.get('name')
                relevant_info.append(f"  • Product: {name}")
            elif 'category' in question_lower and node_type == 'Category':
                name = attrs.get('name')
                relevant_info.append(f"  • Category: {name}")
        
        return "\n".join(relevant_info) if relevant_info else "  Tự động phân tích từ embeddings"
    
    def _calculate_path_coherence(self, path: List[str]) -> float:
        """Tính coherence score của path"""
        if len(path) < 2 or not self.embeddings:
            return 0.0
        
        similarities = []
        for i in range(len(path) - 1):
            if path[i] in self.embeddings and path[i+1] in self.embeddings:
                emb1 = self.embeddings[path[i]].reshape(1, -1)
                emb2 = self.embeddings[path[i+1]].reshape(1, -1)
                sim = cosine_similarity(emb1, emb2)[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

"Lớp Text2CypherGenerator sử dụng các template Cypher được định nghĩa sẵn (cypher_templates)" 
"và chọn template phù hợp dựa trên từ khóa trong câu hỏi và context."
class Text2CypherGenerator:
    """Bước 5: Sinh Cypher query với context"""
    
    def __init__(self):
        self.cypher_templates = {
            'find_products': """
MATCH (p:Product)
WHERE p.productName CONTAINS '{product_name}'
RETURN p.productName, p.unitPrice
""",
            'find_category_products': """
MATCH (p:Product)-[:PART_OF]->(c:Category)
WHERE c.categoryName = '{category_name}'
RETURN p.productName, p.unitPrice, c.categoryName
""",
            'find_customer_orders': """
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)
WHERE cust.companyName CONTAINS '{customer_name}' OR cust.contactName CONTAINS '{customer_name}'
RETURN cust.contactName as CustomerName,
       cust.companyName as Company, 
       o.orderDate as OrderDate, 
       p.productName as ProductName
ORDER BY o.orderDate
"""
        }
    
    def generate_cypher(self, question: str, context: str) -> str:
        """Sinh Cypher query dựa trên câu hỏi và context"""
        question_lower = question.lower()
        
        # Phân tích xem có yêu cầu liệt kê tất cả không
        list_all_keywords = ['list all', 'tất cả', 'all', 'list', 'liệt kê', 'hiển thị']
        is_list_all = any(keyword in question_lower for keyword in list_all_keywords)
        
        if 'sản phẩm' in question_lower or 'product' in question_lower:
            if is_list_all:
                return """
MATCH (p:Product)
RETURN p.productName, p.unitPrice, p.categoryID
ORDER BY p.productName
"""
            elif 'danh mục' in question_lower or 'category' in question_lower:
                # Tìm trong context để lấy category name
                category_name = 'Beverages'  # default
                if 'beverages' in context.lower() or 'đồ uống' in context.lower():
                    category_name = 'Beverages'
                elif 'condiments' in context.lower():
                    category_name = 'Condiments'
                
                return self.cypher_templates['find_category_products'].format(
                    category_name=category_name
                )
            else:
                # Tìm tên sản phẩm trong context hoặc câu hỏi
                product_name = 'Chai'  # default
                if 'chai' in question_lower:
                    product_name = 'Chai'
                elif 'chang' in question_lower:
                    product_name = 'Chang'
                elif 'syrup' in question_lower:
                    product_name = 'Syrup'
                
                return self.cypher_templates['find_products'].format(
                    product_name=product_name
                )
        
        elif 'khách hàng' in question_lower or 'customer' in question_lower:
            if is_list_all:
                return """
MATCH (c:Customer)
RETURN c.contactName as CustomerName,
       c.companyName as Company, 
       c.city as City, 
       c.country as Country,
       c.phone as Phone
ORDER BY c.contactName
"""
            # Kiểm tra xem có hỏi về đồ uống/beverages không
            elif any(drink_word in question_lower for drink_word in ['đồ uống', 'beverage', 'drink']):
                return """
MATCH (cust:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)-[:PART_OF]->(c:Category)
WHERE c.categoryName = 'Beverages'
RETURN DISTINCT cust.contactName as CustomerName, 
       cust.companyName as Company,
       cust.city as City
ORDER BY cust.contactName
"""
            else:
                customer_name = 'Alfreds'  # default
                if 'ana' in question_lower:
                    customer_name = 'Ana'
                elif 'alfreds' in question_lower:
                    customer_name = 'Alfreds'
                
                return self.cypher_templates['find_customer_orders'].format(
                    customer_name=customer_name
                )
        
        elif 'danh mục' in question_lower or 'category' in question_lower:
            if is_list_all:
                return """
MATCH (c:Category)
RETURN c.categoryName, c.description
ORDER BY c.categoryName
"""
        
        elif 'đơn hàng' in question_lower or 'order' in question_lower:
            if is_list_all:
                return """
MATCH (o:Order)
RETURN o.orderID, o.orderDate, o.customerID
ORDER BY o.orderDate DESC
"""
        
        # Default query cho list all
        if is_list_all:
            return """
MATCH (n)
RETURN labels(n) as NodeType, count(n) as Count
"""
        
        # Default query
        return """
MATCH (n)
RETURN n
LIMIT 10
"""

class GNNRAGDemo:
    """Demo chính của Text2Cypher GNN-RAG"""
    
    def __init__(self):
        print("🚀 Khởi tạo Advanced GNN-RAG Demo...")
        self.entity_extractor = EntityExtractor()
        self.graph_builder = GraphBuilder()
        
        # Enhanced path finder với embeddings
        self.path_finder = GNNPathFinder(self.graph_builder.graph, self.graph_builder)
        
        # Enhanced context generator với embeddings
        embeddings = None
        if hasattr(self.path_finder, 'enhanced_embeddings'):
            embeddings = self.path_finder.enhanced_embeddings
        self.context_generator = ContextGenerator(self.graph_builder.graph, embeddings)
        
        self.cypher_generator = Text2CypherGenerator()
        print("✅ Advanced GNN-RAG Demo sẵn sàng!")
    
    def process_question(self, question: str) -> Dict:
        """Xử lý câu hỏi theo luồng GNN-RAG"""
        print(f"\n🔍 Câu hỏi: {question}")
        print("=" * 50)
        
        # Bước 1: Xác định đối tượng
        print("Bước 1: Xác định đối tượng trong câu hỏi...")
        entities = self.entity_extractor.extract_entities(question)
        print(f"Tìm thấy {len(entities)} đối tượng:")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type}) - độ tin cậy: {entity.confidence:.2f}")
        
        # Bước 2: Tìm nodes tương ứng
        print("\nBước 2: Tìm nodes trong knowledge graph...")
        matched_nodes = self.graph_builder.find_entity_nodes(entities)
        print(f"Tìm thấy {len(matched_nodes)} nodes:")
        for node in matched_nodes:
            attrs = self.graph_builder.node_attributes[node]
            display_name = attrs.get('name') or attrs.get('companyName') or 'N/A'
            print(f"  - {node}: {display_name} ({attrs.get('type', 'N/A')})")
        
        # Bước 3: Tìm đường đi bằng Advanced GNN
        print("\nBước 3: Tìm đường đi bằng Advanced GNN...")
        paths = self.path_finder.find_paths(matched_nodes)
        scored_paths = self.path_finder.score_paths(paths)
        print(f"Tìm thấy {len(scored_paths)} đường đi:")
        for path, score in scored_paths[:3]:
            print(f"  - Điểm: {score:.3f} | Đường đi: {' -> '.join(path)}")
        
        # Bước 4: Tạo enhanced context
        print("\nBước 4: Tạo enhanced context từ đường đi...")
        best_paths = [path for path, score in scored_paths[:3]]
        context = self.context_generator.generate_context(best_paths, question)
        
        # Nếu không có đường đi, tạo context từ matched nodes
        if not best_paths and matched_nodes:
            context_parts = ["=== THÔNG TIN TỪ KNOWLEDGE GRAPH ===\n"]
            context_parts.append("Các nodes liên quan:")
            for node in matched_nodes[:5]:  # Chỉ lấy 5 nodes đầu
                attrs = self.graph_builder.node_attributes[node]
                node_type = attrs.get('type', 'Unknown')
                node_name = attrs.get('name') or attrs.get('companyName') or node
                context_parts.append(f"- {node_type}: {node_name}")
            context = "\n".join(context_parts)
        
        print("Context đã tạo:")
        print(context)
        
        # Bước 5: Sinh Cypher query
        print("\nBước 5: Sinh Cypher query...")
        cypher_query = self.cypher_generator.generate_cypher(question, context)
        print("Cypher query:")
        print(cypher_query)
        
        # Thông tin về embeddings
        embeddings_info = {}
        if ADVANCED_AVAILABLE and hasattr(self.path_finder, 'enhanced_embeddings'):
            embeddings_info = {
                'text_embedding_dim': self.path_finder.sentence_model.get_sentence_embedding_dimension(),
                'enhanced_embedding_dim': list(self.path_finder.enhanced_embeddings.values())[0].shape[0],
                'total_nodes': len(self.path_finder.enhanced_embeddings),
                'gcn_trained': True
            }
        else:
            embeddings_info = {
                'text_embedding_dim': 0,
                'enhanced_embedding_dim': 64,  # fallback
                'total_nodes': len(matched_nodes),
                'gcn_trained': False
            }
        
        print(f"\n📊 Embedding Info:")
        print(f"  Text embedding dim: {embeddings_info['text_embedding_dim']}")
        print(f"  Enhanced embedding dim: {embeddings_info['enhanced_embedding_dim']}")
        print(f"  GCN trained: {embeddings_info['gcn_trained']}")
        
        return {
            'question': question,
            'entities': entities,
            'matched_nodes': matched_nodes,
            'paths': scored_paths,
            'context': context,
            'cypher_query': cypher_query,
            'embeddings_info': embeddings_info
        }

def main():
    """Hàm chính để chạy demo"""
    print("🚀 DEMO TEXT2CYPHER GNN-RAG")
    print("=" * 50)
    print("Demo này minh họa luồng xử lý từ câu hỏi tiếng Việt/Anh đến Cypher query")
    print("Sử dụng knowledge graph về hệ thống quản lý đơn hàng\n")
    
    # Khởi tạo demo
    demo = GNNRAGDemo()
    
    # Các câu hỏi mẫu
    sample_questions = [
        "Tìm thông tin về sản phẩm Chai",
        "Khách hàng nào đã mua đồ uống?",
        "Sản phẩm nào thuộc danh mục Beverages?",
        "Đơn hàng của khách hàng Alfreds"
    ]
    
    while True:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Chọn câu hỏi mẫu")
        print("2. Nhập câu hỏi tùy chỉnh")
        print("3. Xem cấu trúc knowledge graph")
        print("4. Thoát")
        
        choice = input("\nChọn lựa (1-4): ").strip()
        
        if choice == '1':
            print("\nCác câu hỏi mẫu:")
            for i, q in enumerate(sample_questions, 1):
                print(f"{i}. {q}")
            
            try:
                q_choice = int(input("\nChọn câu hỏi (1-4): ")) - 1
                if 0 <= q_choice < len(sample_questions):
                    demo.process_question(sample_questions[q_choice])
                else:
                    print("Lựa chọn không hợp lệ!")
            except ValueError:
                print("Vui lòng nhập số!")
        
        elif choice == '2':
            question = input("\nNhập câu hỏi của bạn: ").strip()
            if question:
                demo.process_question(question)
            else:
                print("Câu hỏi không được để trống!")
        
        elif choice == '3':
            print("\n📊 Cấu trúc Knowledge Graph:")
            graph = demo.graph_builder.graph
            print(f"Số nodes: {graph.number_of_nodes()}")
            print(f"Số edges: {graph.number_of_edges()}")
            
            print("\nCác loại nodes:")
            node_types = {}
            for node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            for node_type, count in node_types.items():
                print(f"  - {node_type}: {count} nodes")
            
            print("\nCác loại relationships:")
            rel_types = {}
            for edge in graph.edges():
                rel_type = graph.edges[edge].get('relation', 'Unknown')
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            for rel_type, count in rel_types.items():
                print(f"  - {rel_type}: {count} relationships")
        
        elif choice == '4':
            print("Cảm ơn bạn đã sử dụng demo! 👋")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 