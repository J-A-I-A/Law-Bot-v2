import json
import logging
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
database = Pinecone(api_key=api_key)
index_laws = database.Index("multilingual-e5-large")

def relevant_info(question: str) -> str:
    logging.info(f"Retrieving relevant law info for question: {question}")
    embeddings = database.inference.embed(
        "multilingual-e5-large",
        inputs=question,
        parameters={"input_type": "query"},
    )
    search_results = index_laws.query(
        vector=embeddings[0]['values'],
        namespace="laws",
        top_k=100,
        include_values=False,
        include_metadata=True,
    )
    law_documents = [
        {"id": match["id"], "text": match["metadata"]["text"]}
        for match in search_results['matches']
        if 'text' in match['metadata']
    ]
    logging.info(f"Pinecone query returned {len(law_documents)} candidate document(s).")
    ranked_response = database.inference.rerank(
        model="bge-reranker-v2-m3",
        query=question,
        documents=law_documents,
        top_n=30,
        return_documents=True,
    )

    ranked_documents = [{"id": r.document["id"], "text": r.document["text"]} for r in ranked_response.data if r.document]

    # Prepare the law content
    law_content = [{"id": doc['id'], "text": doc['text']} for doc in ranked_documents]
    logging.info(f"Returning {len(law_content)} reranked document(s) to the bot.")

    law_json = json.dumps(law_content, ensure_ascii=False)

    return law_json




