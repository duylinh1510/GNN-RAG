#!/usr/bin/env python3
"""
Text2Cypher RAG Demo - Console Version
Chuyển đổi câu hỏi tiếng tự nhiên thành câu truy vấn Neo4j Cypher sử dụng RAG
"""

import os
import sys
from typing import Dict, List, Any
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

class Text2CypherRAG:
    def __init__(self):
        # Initialize Ollama with a free model
        self.llm = OllamaLLM(model="llama3.2")
        
        # Initialize embeddings (free Hugging Face model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Example queries for demonstration (định nghĩa trước để sử dụng trong _load_schema_knowledge)
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
            },
            {
                "question": "Show products with low stock",
                "cypher": "MATCH (p:Product) WHERE p.unitsInStock < 10 RETURN p.productName, p.unitsInStock"
            },
            {
                "question": "Find expensive products over 50",
                "cypher": "MATCH (p:Product) WHERE p.unitPrice > 50 RETURN p.productName, p.unitPrice ORDER BY p.unitPrice DESC"
            },
            {
                "question": "Which customers bought products from suppliers in UK?",
                "cypher": "MATCH (c:Customer)-[:PURCHASED]->(o:Order)-[:ORDERS]->(p:Product)<-[:SUPPLIES]-(s:Supplier) WHERE s.country = 'UK' RETURN DISTINCT c.companyName, s.companyName"
            },
            {
                "question": "Show products cheaper than 20 in Beverages category",
                "cypher": "MATCH (p:Product)-[:PART_OF]->(c:Category) WHERE c.categoryName = 'Beverages' AND p.unitPrice < 20 RETURN p.productName, p.unitPrice"
            }
        ]
        
        # Load schema and create knowledge base
        self.schema_docs = self._load_schema_knowledge()
        self.vector_store = self._create_vector_store()
    
    def _load_schema_knowledge(self) -> List[Document]:
        """Load and process schema information into documents"""
        schema_docs = []
        
        # Load schema file
        with open('schema.txt', 'r', encoding='utf-8') as f:
            schema_content = f.read()
        
        # Node information
        nodes_info = """
        Node Types and Properties:
        
        1. Product: Contains product information
           - productName, quantityPerUnit, unitsOnOrder, supplierID, productID
           - discontinued, categoryID, reorderLevel, unitsInStock, unitPrice
        
        2. Category: Product categories
           - categoryID, categoryName, description, picture
        
        3. Supplier: Product suppliers
           - supplierID, companyName, contactName, phone, address, city, country
        
        4. Customer: Customer information
           - customerID, companyName, contactName, phone, address, city, country
        
        5. Order: Order information
           - orderID, orderDate, requiredDate, shippedDate, freight, shipName, shipAddress
        """
        
        relationships_info = """
        Relationships:
        
        1. (:Product)-[:PART_OF]->(:Category) - Products belong to categories
        2. (:Supplier)-[:SUPPLIES]->(:Product) - Suppliers supply products
        3. (:Customer)-[:PURCHASED]->(:Order) - Customers purchase orders
        4. (:Order)-[:ORDERS]->(:Product) - Orders contain products
        """
        
        cypher_patterns = """
        Common Cypher Query Patterns:
        
        1. Find all products in a category:
           MATCH (p:Product)-[:PART_OF]->(c:Category) WHERE c.categoryName = 'CategoryName' RETURN p
        
        2. Get supplier information for a product:
           MATCH (s:Supplier)-[:SUPPLIES]->(p:Product) WHERE p.productName = 'ProductName' RETURN s
        
        3. Find customer orders:
           MATCH (c:Customer)-[:PURCHASED]->(o:Order) WHERE c.companyName = 'CompanyName' RETURN o
        
        4. Get products in an order:
           MATCH (o:Order)-[:ORDERS]->(p:Product) WHERE o.orderID = 'OrderID' RETURN p
        
        5. Filter by properties:
           MATCH (p:Product) WHERE p.unitPrice > 50 RETURN p.productName, p.unitPrice
        """
        
        # Create documents
        schema_docs.extend([
            Document(page_content=schema_content, metadata={"type": "schema"}),
            Document(page_content=nodes_info, metadata={"type": "nodes"}),
            Document(page_content=relationships_info, metadata={"type": "relationships"}),
            Document(page_content=cypher_patterns, metadata={"type": "patterns"})
        ])
        
        # Add example queries
        for example in self.example_queries:
            doc_content = f"Question: {example['question']}\nCypher: {example['cypher']}"
            schema_docs.append(Document(page_content=doc_content, metadata={"type": "example"}))
        
        return schema_docs
    
    def _create_vector_store(self):
        """Create FAISS vector store from schema documents"""
        return FAISS.from_documents(self.schema_docs, self.embeddings)
    
    def _get_relevant_context(self, question: str, k: int = 3) -> str:
        """Retrieve relevant context for the question"""
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def _validate_cypher_query(self, cypher_query: str) -> str:
        """Validate Cypher query against schema and return corrected query or error"""
        # Valid node labels from schema
        valid_nodes = {'Product', 'Category', 'Supplier', 'Customer', 'Order'}
        
        # Valid relationship types from schema  
        valid_relationships = {'PART_OF', 'SUPPLIES', 'PURCHASED', 'ORDERS'}
        
        # Convert to uppercase for case-insensitive comparison
        query_upper = cypher_query.upper()
        
        # Check for invalid node labels
        import re
        # Find node patterns like (n:NodeLabel) or (:NodeLabel) - chỉ sau dấu :
        node_pattern = r'\(\w*:([A-Z][A-Z_a-z]*)\)'
        found_nodes = re.findall(node_pattern, query_upper)
        
        for node in found_nodes:
            if node not in [n.upper() for n in valid_nodes]:
                return f"ERROR: Node '{node}' không tồn tại trong schema. Các node hợp lệ: {', '.join(valid_nodes)}"
        
        # Check for invalid relationships  
        rel_pattern = r'\[:([A-Z_]+)\]'
        found_rels = re.findall(rel_pattern, query_upper)
        
        for rel in found_rels:
            if rel not in [r.upper() for r in valid_relationships]:
                return f"ERROR: Relationship '{rel}' không tồn tại trong schema. Các relationship hợp lệ: {', '.join(valid_relationships)}"
        
        return cypher_query
    
    def generate_cypher(self, question: str) -> str:
        """Generate Cypher query from natural language question"""
        # Get relevant context
        context = self._get_relevant_context(question)
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert in converting natural language questions to Neo4j Cypher queries.

Given the following database schema and context:
{context}

Convert this natural language question to a Cypher query:
Question: {question}

CRITICAL VALIDATION RULES:
1. ONLY use these exact node labels: Product, Category, Supplier, Customer, Order
2. ONLY use these exact relationships: PART_OF, SUPPLIES, PURCHASED, ORDERS
3. If the question asks about entities NOT in the schema (like employees, users, departments, people), return: "ERROR: Entities not found in schema"
4. Do NOT guess or substitute - if unsure, return error message
5. Use proper Neo4j syntax
6. Include necessary WHERE clauses for filtering
7. Use RETURN to specify what data to retrieve

ENTITY MAPPING (be strict):
- products → Product ✓
- categories → Category ✓  
- suppliers → Supplier ✓
- customers → Customer ✓
- orders → Order ✓
- employees → NOT EXIST → ERROR ✗
- users → NOT EXIST → ERROR ✗
- departments → NOT EXIST → ERROR ✗
- people → NOT EXIST → ERROR ✗
- companies → Use Supplier or Customer (clarify in context)

If the question cannot be mapped to valid schema entities, return:
"ERROR: Cannot map question entities to schema. Available entities: Product, Category, Supplier, Customer, Order"

Cypher Query:
"""
        )
        
        # Generate query
        prompt = prompt_template.format(context=context, question=question)
        cypher_query = self.llm.invoke(prompt)
        
        # Clean up the response
        cypher_query = cypher_query.strip()
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        # Check if LLM already returned an error message
        if cypher_query.startswith("ERROR:"):
            return cypher_query
        
        # Validate the generated query
        validated_query = self._validate_cypher_query(cypher_query)
        
        return validated_query

def main():
    print("🔍 Text2Cypher RAG Demo (Console Version)")
    print("=" * 50)
    
    try:
        print("Đang khởi tạo hệ thống RAG...")
        rag_system = Text2CypherRAG()
        print("✅ Khởi tạo hệ thống thành công!\n")
        
        # Show example queries
        print("💡 Các câu hỏi mẫu bạn có thể hỏi:")
        for i, example in enumerate(rag_system.example_queries, 1):
            print(f"{i}. {example['question']}")
        print()
        
        while True:
            try:
                question = input("Nhập câu hỏi của bạn (hoặc 'quit' để thoát): ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', 'thoát']:
                    print("Tạm biệt! 👋")
                    break
                
                if not question:
                    continue
                
                print("\n🔄 Đang tạo câu truy vấn Cypher...")
                cypher_query = rag_system.generate_cypher(question)
                
                print("\n📋 Câu truy vấn Cypher đã tạo:")
                print("-" * 40)
                print(cypher_query)
                print("-" * 40)
                
                # Ask if user wants to see context
                show_context = input("\nHiển thị ngữ cảnh được truy xuất? (y/n): ").strip().lower()
                if show_context in ['y', 'yes', 'có']:
                    context = rag_system._get_relevant_context(question)
                    print("\n📚 Ngữ cảnh được truy xuất:")
                    print("-" * 40)
                    print(context)
                    print("-" * 40)
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nTạm biệt! 👋")
                break
            except Exception as e:
                print(f"❌ Lỗi: {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Không thể khởi tạo hệ thống: {str(e)}")
        print("\n🔧 Hướng dẫn khắc phục:")
        print("1. Đảm bảo Ollama đang chạy: ollama serve")
        print("2. Cài đặt model llama3.2: ollama pull llama3.2")
        print("3. Cài đặt dependencies: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 