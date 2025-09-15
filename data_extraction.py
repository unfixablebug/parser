def process_text_and_tables(chunks):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq

    from dotenv import load_dotenv
    load_dotenv()

    texts = [chunk for chunk in chunks if chunk.category != "Table" and chunk.text and chunk.text.strip()]
    temp_tables = [chunk for chunk in chunks if chunk.category == "Table" ]
    tables = [table.metadata.text_as_html for table in temp_tables if hasattr(table.metadata, 'text_as_html') and table.metadata.text_as_html]

    prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    return text_summaries, table_summaries

def process_images(chunks):
    import base64
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    image_summaries = chain.batch(images)
    return image_summaries


# def create_chunks(report = "./inputs/report0.pdf"):
report = "./inputs/report0.pdf"
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Table, CompositeElement
# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_pdf(
    filename=report,
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables
    # for image extractions
    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    extract_image_block_to_payload=True,
    chunking_strategy="by_title", 
    max_characters=10000, 
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
)
# images = process_images(chunks)
texts, tables = process_text_and_tables(chunks)
# print(images[0])
print(texts[0])
print(tables[0])