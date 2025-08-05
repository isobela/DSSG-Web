import argparse
import os
from dotenv import load_dotenv

from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

PROMPT_TEMPLATE = """
You are a domain expert in mentorship. The context that you are receiving comes from multiple reports containing mentorship findings,
information,and statistical insights that have all been published by Mentor Canada. Because these chunks retrieved from the sources may be 
disjoint, specify the context as much as possible in your response. For example, if asked about what are good qualities of a mentor, and if
you come across a chunk that addresses this question specific to newcomer youth, include this context (e.g., "For newcomer youth specifically, 
a good mentor may ..."). However, try to diversify the response and includes different groups in your response.

INSTRUCTION:

You must always include a section at the end of the response titled "References." This section should list a maximum of 5 unique source documents,
and all sources in this format: (Source: rag/processed_pdfs/FILENAME.pdf:PAGE_NUMBER:POSITION, page PAGE_NUMBER) must be listed.

- Each source should appear only once in the References section.
- If multiple pieces of information come from the same PDF, list all relevant page numbers on the same line.
- Format page numbers as integers, separated by commas.

Example:
If these two source chunks are used:
- rag/processed_pdfs/SRDC_Final_Report_ocr.pdf:55:0
- rag/processed_pdfs/SRDC_Final_Report_ocr.pdf:22:0

Then the reference should appear as:

State of Mentoring Youth Survey Report: December 2020, page 55, 22

  
You will be given a list of source lines in the following format:

(Source: rag/processed_pdfs/FILENAME.pdf:PAGE_NUMBER:POSITION, page PAGE_NUMBER)

Your task is to extract:
1. The original filename from the path (e.g., Becoming_a_Better_Mentor_ocr.pdf)
2. The integer page number (e.g., 89 from "page 89.0"), THERE SHOULD NEVER BE A DECIMAL (.) DESCRIBING THE PAGE NUMBER
3. Replace the filename using the following readable name mapping:

Use this mapping of the original PDF names (left of the colon) and their more readable names (right of the colon):

       "rag/processed_pdfs/Becoming_a_Better_Mentor_ocr.pdf":"Becoming a Better Mentor: Strategies to be There for Young People",
        "rag/processed_pdfs/Confidential_Draft_SRDC_Report_ocr.pdf":"Unlocking Doors: Research on Mentoring to Strengthen Skills & Support Career Pathways for Racialized young Adults",
        "rag/processed_pdfs/Effective_Elements_For_Mentorship_ocr.pdf":"ELEMENTS OF EFFECTIVE PRACTICE FOR MENTORING: A Guide for Program Development and Improvement",
        "rag/processed_pdfs/Mapping_the_Gap_Report_ocr.pdf":"Mapping the Mentoring Gap Report: The State of Mentoring in Canada May 2021",
        "rag/processed_pdfs/"MENTOR_The_Mentoring_Effect_Full_Report_ocr.pdf":"The Mentoring Effect: Young People's Perspectives on the Outcomes and Availability of Mentoring",
        "rag/processed_pdfs/Newcomer_Mentoring Effect_Brief_ocr.pdf":"The Mentoring Effect: Newcomer Youth", 
        "rag/processed_pdfs/SRDC_Final_Report_ocr.pdf":"State of Mentoring Youth Survey Report: December 2020",
        "rag/processed_pdfs/SRDC_Final_RTP_Report_Dec15_FINAL_ocr.pdf":"Raising the Profile Report",
        "rag/processed_pdfs/Who-Mentored-You_ocr.pdf":"Who Mentored You 2023" 

Using this mapping, transform the source lines into a “References” section like this:

EXAMPLE INPUT:

(Source: rag/processed_pdfs/Becoming_a_Better_Mentor_ocr.pdf:89:0, page 89.0)
(Source: rag/processed_pdfs/Who-Mentored-You_ocr.pdf:2:0, page 2.0)
(Source: rag/processed_pdfs/Mapping_the_Gap_Report_ocr.pdf:3:0, page 3.0)
(Source: rag/processed_pdfs/MENTOR_The_Mentoring_Effect_Full_Report_ocr.pdf:8:0, page 8.0)
(Source: rag/processed_pdfs/Newcomer_Mentoring Effect_Brief_ocr.pdf:20:0, page 20.0)

EXPECTED OUTPUT:

References: 
- Becoming a Better Mentor: Strategies to be There for Young People, page 89
- Who Mentored You 2023, page 2
- Mapping the Mentoring Gap Report: The State of Mentoring in Canada May 2021, page 3
- Young People's Perspectives on the Outcomes and Availability of Mentoring, page 8
- The Mentoring Effect: Newcomer Youth, page 20

Maintain a conversational tone while delivering sensitive information with care and nuance. Avoid run-on sentences and excessive lists. 
The response should flow smoothly while retaining a high level of detail.

Answer the question based only on the following context:

{context}

--------------------------------

Answer the question based on the above context: {question}
"""
# get variables from .env file
load_dotenv()

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text:str):

    # load the database with the embeddings
    embedding_function = get_embedding_function()

    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    db = PineconeVectorStore(index=index, embedding=embedding_function, text_key="text")


    # search the database
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("No relevant documents retrieved for this query.")
    else:
        for doc, score in results:
            print(f"Score: {score}, ID: {doc.metadata.get('id')}, Content preview: {doc.page_content[:300]}")

    PDF_NAME_MAP = {
        "Becoming_a_Better_Mentor_ocr.pdf":"Becoming a Better Mentor: Strategies to be There for Young People",
        "Confidential_Draft_SRDC_Report_ocr.pdf":"Unlocking Doors: Research on Mentoring to Strengthen Skills & Support Career Pathways for Racialized young Adults",
        "Effective_Elements_For_Mentorship_ocr.pdf":"ELEMENTS OF EFFECTIVE PRACTICE FOR MENTORING: A Guide for Program Development and Improvement",
        "Mapping_the_Gap_Report_ocr.pdf":"Mapping the Mentoring Gap Report: The State of Mentoring in Canada May 2021",
        "MENTOR_The_Mentoring_Effect_Full_Report_ocr.pdf":"The Mentoring Effect: Young People's Perspectives on the Outcomes and Availability of Mentoring",
        "Newcomer_Mentoring Effect_Brief_ocr.pdf":"The Mentoring Effect: Newcomer Youth", 
        "SRDC_Final_Report_ocr.pdf":"State of Mentoring Youth Survey Report: December 2020",
        "SRDC_Final_RTP_Report_Dec15_FINAL_ocr.pdf":"Raising the Profile Report",
        "Who-Mentored-You_ocr.pdf":"Who Mentored You 2023" 
    }

    context_text = "\n\n---\n\n".join([
    f"{doc.page_content}\n(Source: {PDF_NAME_MAP.get(doc.metadata.get('id', 'Unknown'), doc.metadata.get('id', 'Unknown'))}, page {doc.metadata.get('page', 'N/A')})"
    for doc, _ in results
])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    references = sorted({
        f"{PDF_NAME_MAP.get(os.path.basename(doc.metadata.get('id', 'Unknown')), 'Unknown')}, page {int(float(doc.metadata.get('page', 0)))}"
        for doc, _ in results
    })

    # Combine the model response with the cleaned references last reference section
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

