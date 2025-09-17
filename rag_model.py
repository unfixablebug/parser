import json
import time
from typing import List
from openai import RateLimitError

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def create_partitions(report):
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title

    # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    elements = partition_pdf(
        filename=report,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables
        # for image extractions
        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        extract_image_block_to_payload=True
    )
    return elements
def create_chunks(elements):
    from unstructured.chunking.title import chunk_by_title
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    return chunks
def create_summaries(chunks):
    langchain_documents = []

    for i, chunk in enumerate(chunks):
        content_data = separate_chunk_by_type(chunk)
        if content_data['tables'] or content_data['images']:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
            except Exception as e:
                enhanced_content = content_data['text']
        else:
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    return langchain_documents
def separate_chunk_by_type(chunk):
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data
def create_ai_enhanced_summary(text, tables, images):
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        try: 
            response = llm.invoke([message])
        except RateLimitError:
            time.sleep(60)
            response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary
def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )  
    return vectorstore
def run_complete_ingestion_pipeline(report):
    # Step 1: Partition
    elements = create_partitions(report)
    
    # Step 2: Chunk
    chunks = create_chunks(elements)
    
    # Step 3: AI Summarisation
    summarised_chunks = create_summaries(chunks)
    
    # Step 4: Vector Store
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db")
    
    return db
def query_and_answer_generation(db, query):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""
        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        try:
            response = llm.invoke([message])
        except RateLimitError:
            time.sleep(60)
            response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."
def rag_model(report):
    db = run_complete_ingestion_pipeline(report)
    with open('./prompts.txt', 'r') as file:    
        for line in file:
            if line.strip():
                print(query_and_answer_generation(db, line.strip()))
rag_model("./inputs/report.pdf")