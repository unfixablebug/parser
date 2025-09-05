from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Table, CompositeElement
from IPython.display import display, HTML, Image
import base64
import os

for number in range(3):
    file_path = f"./inputs/report{number}.pdf"
    os.makedirs(f"./outputs/report{number}", exist_ok=True)
    # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables
        # for image extractions
        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        extract_image_block_to_payload=True,
        # chunking_strategy=None,
    # )
    # text_chunks = partition_pdf(
        # filename=file_path,
        # infer_table_structure=True,     
        # strategy="hi_res",                

        chunking_strategy="by_title", 
        max_characters=10000, 
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64
    
    images = get_images_base64(chunks)
    texts = [chunk for chunk in chunks if chunk.category != "Table" and chunk.text and chunk.text.strip()]
    temp_tables = [chunk for chunk in chunks if chunk.category == "Table" ]
    tables = [table.metadata.text_as_html for table in temp_tables if hasattr(table.metadata, 'text_as_html') and table.metadata.text_as_html]

    # exporting the image
    for i, image_data in enumerate(images):
        img_data = base64.b64decode(image_data)
        with open(f"./outputs/report{number}/image_{i+1}.png", "wb") as f:
            f.write(img_data)

    # exporting the table as html
    for i, table in enumerate(tables):
        with open(f"./outputs/report{number}/table_{i+1}.html", "w", encoding="utf-8") as f:
            f.write(table)

    # exporting texts
    with open(f"./outputs/report{number}/extracted_text.txt", "w", encoding="utf-8") as f:
        for chunk in texts:
            f.write(chunk.text)