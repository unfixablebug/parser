import json 
import time 
import io 
from typing import List 
from openai import RateLimitError 
from dotenv import load_dotenv 
 
from unstructured.partition.pdf import partition_pdf 
from unstructured.chunking.title import chunk_by_title 
 
from langchain_core.documents import Document 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.messages import HumanMessage 
 
from googleapiclient.discovery import build 
from google.oauth2 import service_account 
from googleapiclient.http import MediaIoBaseDownload 
 
load_dotenv() 
 
def create_partitions(report): 
    print("Partitioning document...") 
    if isinstance(report, str): 
        elements = partition_pdf( 
            filename=report, 
            infer_table_structure=True, 
            strategy="hi_res", 
            extract_image_block_types=["Image"], 
            extract_image_block_to_payload=True 
        ) 
    else: 
        elements = partition_pdf( 
            file=report, 
            infer_table_structure=True, 
            strategy="hi_res", 
            extract_image_block_types=["Image"], 
            extract_image_block_to_payload=True 
        ) 
    print(f"Partitioned into {len(elements)} elements.") 
    return elements 
 
def create_chunks(elements): 
    print("Chunking content...") 
    chunks = chunk_by_title( 
        elements, 
        max_characters=3000, 
        new_after_n_chars=2400, 
        combine_text_under_n_chars=500 
    ) 
    print(f"Created {len(chunks)} chunks.") 
    return chunks 
 
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
            elif element_type == 'Image': 
                if hasattr(element.metadata, 'image_base64'): 
                    content_data['types'].append('image') 
                    content_data['images'].append(element.metadata.image_base64) 
    content_data['types'] = list(set(content_data['types'])) 
    return content_data 
 
def create_ai_enhanced_summary(text, tables, images): 
    try: 
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        prompt_text = f"""You are creating a searchable description for document content retrieval.\n\nTEXT CONTENT:\n{text}\n\n""" 
        if tables: 
            prompt_text += "TABLES:\n" 
            for i, table in enumerate(tables): 
                prompt_text += f"Table {i+1}:\n{table}\n\n" 
            prompt_text += ( 
                "YOUR TASK:\nGenerate a comprehensive, searchable description that covers:\n" 
                "1. Key facts, numbers, and data points from text and tables\n" 
                "2. Main topics and concepts discussed\n" 
                "3. Questions this content could answer\n" 
                "4. Visual content analysis\n" 
                "5. Alternative search terms\n\n" 
                "SEARCHABLE DESCRIPTION:" 
            ) 
        message_content = [{"type": "text", "text": prompt_text}] 
        for image_base64 in images: 
            message_content.append({ 
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"} 
            }) 
        message = HumanMessage(content=message_content) 
        try: 
            response = llm.invoke([message]) 
        except RateLimitError: 
            time.sleep(60) 
            response = llm.invoke([message]) 
        return response.content 
    except Exception as e: 
        summary = f"{text[:300]}..." 
        if tables: 
            summary += f" [Contains {len(tables)} table(s)]" 
        if images: 
            summary += f" [Contains {len(images)} image(s)]" 
        return summary 
 
def create_summaries(chunks): 
    print("Summarizing chunks with GPT-4o...") 
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
            except Exception: 
                enhanced_content = content_data['text'] 
        else: 
            enhanced_content = content_data['text'] 
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
    print(f"Created {len(langchain_documents)} summarized documents.") 
    return langchain_documents 
 
def create_vector_store(documents, persist_directory="dbv2/chroma_db"): 
    print("Storing in ChromaDB...") 
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    vectorstore = Chroma.from_documents( 
        documents=documents, 
        embedding=embedding_model, 
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} 
    ) 
    print("Vector store created.") 
    return vectorstore 
 
def run_complete_ingestion_pipeline(report, allow_drive_download=False): 
    if isinstance(report, str) and report.startswith("drive:"): 
        file_id = report.replace("drive:", "") 
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] 
        SERVICE_ACCOUNT_FILE = 'keys/esg-agent-key.json' 
        credentials = service_account.Credentials.from_service_account_file( 
            SERVICE_ACCOUNT_FILE, scopes=SCOPES 
        ) 
        drive_service = build('drive', 'v3', credentials=credentials) 
 
        file_metadata = drive_service.files().get(fileId=file_id).execute() 
        mime_type = file_metadata.get("mimeType", "") 
        print(f"Successfully accessed Google Drive file: {file_metadata.get('name')} ({mime_type})") 
 
        # Handle folders 
        if mime_type == "application/vnd.google-apps.folder": 
            print("Detected a Google Drive folder. Listing contents...") 
            query = f"'{file_id}' in parents and trashed = false" 
            results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute() 
            files = results.get("files", []) 
 
            if not files: 
                print("No files found in the folder.") 
                return None 
 
            for i, file in enumerate(files): 
                print(f"report{i}: {file['name']} ({file['mimeType']})") 
                # Optional: ingest each file if it's a PDF and downloading is allowed 
                if allow_drive_download and file["mimeType"] == "application/pdf": 
                    print(f"Ingesting report{i}: {file['name']}") 
                    sub_report = f"drive:{file['id']}" 
                    run_complete_ingestion_pipeline(sub_report, allow_drive_download=True) 
            return None  # Skip further processing for folder itself 
 
        # Handle individual files 
        if not allow_drive_download: 
            print(f"Skipping download from Google Drive: {report}") 
            return None 
 
        print(f"Streaming document from Google Drive: {file_id}") 
        if mime_type == "application/pdf": 
            request = drive_service.files().get_media(fileId=file_id) 
        else: 
            print("File is not a native PDF. Attempting export to PDF...") 
            request = drive_service.files().export_media(fileId=file_id, mimeType='application/pdf') 
 
        fh = io.BytesIO() 
        downloader = MediaIoBaseDownload(fh, request) 
        done = False 
        while not done: 
            status, done = downloader.next_chunk() 
        fh.seek(0) 
        report = fh 
 
    elements = create_partitions(report) 
    chunks = create_chunks(elements) 
    summarised_chunks = create_summaries(chunks) 
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db") 
    print("Ingestion complete.") 
    return db 
 
def query_and_answer_generation(db, query): 
    print(f"\nQuery: {query}") 
    retriever = db.as_retriever(search_kwargs={"k": 3}) 
    chunks = retriever.invoke(query) 
    try: 
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n" 
        for i, chunk in enumerate(chunks): 
            prompt_text += f"--- Document {i+1} ---\n" 
            if "original_content" in chunk.metadata: 
                original_data = json.loads(chunk.metadata["original_content"]) 
                raw_text = original_data.get("raw_text", "") 
                if raw_text: 
                    prompt_text += f"TEXT:\n{raw_text}\n\n" 
                tables_html = original_data.get("tables_html", []) 
                if tables_html: 
                    prompt_text += "TABLES:\n" 
                    for j, table in enumerate(tables_html): 
                        prompt_text += f"Table {j+1}:\n{table}\n\n" 
            prompt_text += "\n" 
        prompt_text += ( "Please provide a clear, comprehensive answer using the text, tables, and images above. " "If the documents don't contain sufficient information to answer the question, say " "\"I don't have enough information to answer that question based on the provided documents.\"\n\n" 
            "ANSWER:" 
        ) 
        message_content = [{"type": "text", "text": prompt_text}] 
        for chunk in chunks: 
            if "original_content" in chunk.metadata: 
                original_data = json.loads(chunk.metadata["original_content"]) 
                images_base64 = original_data.get("images_base64", []) 
                for image_base64 in images_base64: 
                    message_content.append({ 
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"} 
                    }) 
        message = HumanMessage(content=message_content) 
        response = llm.invoke([message]) 
        return response.content 
    except Exception as e: 
        print(f"Answer generation failed: {e}") 
        return "Sorry, I encountered an error while generating the answer." 
 
def rag_model(report): 
    print(f"\nStarting ingestion for: {report}") 
    db = run_complete_ingestion_pipeline(report) 
    print(f"\nAnswering queries for: {report}") 
    try: 
        with open('./prompts.txt', 'r') as file: 
            for line in file: 
                query = line.strip() 
                if query: 
                    answer = query_and_answer_generation(db, query) 
                    print(f"\nQuery: {query}\nAnswer:\n{answer}") 
    except FileNotFoundError: 
        print("prompts.txt not found. Please create the file with one query per line.") 
 
if __name__ == "__main__": 
    print("Starting ESG RAG pipeline...") 
    rag_model("drive:1WoKxRCPNVIdLQfzmITZFnSqyVTJCybkr")