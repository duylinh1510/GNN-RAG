import os
from typing import Dict, List, Any
import streamlit as st
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
        
        # Clean up the response
        cypher_query = cypher_query.strip()
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        return cypher_query

def main():
    st.set_page_config(page_title="Text2Cypher RAG Demo", page_icon="🔍")
    
    st.title("🔍 Text2Cypher RAG Demo")
    st.markdown("Convert natural language questions to Neo4j Cypher queries using RAG")
    
    # Initialize the RAG system
    @st.cache_resource
    def load_rag_system():
        return Text2CypherRAG()
    
    try:
        rag_system = load_rag_system()
        
        # Sidebar with schema information
        with st.sidebar:
            st.header("📊 Database Schema")
            st.markdown("""
            **Nodes:**
            - Product
            - Category  
            - Supplier
            - Customer
            - Order
            
            **Relationships:**
            - Product → Category (PART_OF)
            - Supplier → Product (SUPPLIES)
            - Customer → Order (PURCHASED)
            - Order → Product (ORDERS)
            """)
            
            st.header("💡 Example Questions")
            for i, example in enumerate(rag_system.example_queries):
                if st.button(f"Example {i+1}", key=f"ex_{i}"):
                    st.session_state.question = example["question"]
        
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
            with st.spinner("Generating Cypher query..."):
                try:
                    cypher_query = rag_system.generate_cypher(question)
                    
                    st.header("Generated Cypher Query")
                    st.code(cypher_query, language="cypher")
                    
                    # Show relevant context used
                    with st.expander("View Retrieved Context"):
                        context = rag_system._get_relevant_context(question)
                        st.text(context)
                    
                except Exception as e:
                    st.error(f"Error generating query: {str(e)}")
        
        # Additional information
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Your question is embedded using sentence transformers
        2. Relevant schema information is retrieved using similarity search
        3. The context and question are sent to Ollama (llama3.2)
        4. The model generates a Cypher query based on the schema
        
        **Note:** Make sure Ollama is running with llama3.2 model installed:
        ```bash
        ollama pull llama3.2
        ollama serve
        ```
        """)
        
    except Exception as e:
        st.error(f"Failed to initialize the system: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        1. Ensure Ollama is installed and running
        2. Pull the llama3.2 model: `ollama pull llama3.2`
        3. Install required dependencies: `pip install -r requirements.txt`
        """)

if __name__ == "__main__":
    main() 