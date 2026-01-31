import os
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI

class MedicalRAG:
    def __init__(self, api_key_pinecone, env_pinecone, api_key_openai):
        # Initialize Vector DB
        pinecone.init(api_key=api_key_pinecone, environment=env_pinecone)
        self.index_name = "parkinson-mri-index"
        self.index = pinecone.Index(self.index_name)
        self.llm = OpenAI(temperature=0, openai_api_key=api_key_openai)

    def search_similar_cases(self, embedding_vector, top_k=3):
        """
        Finds past patient cases similar to the current MRI scan.
        """
        results = self.index.query(
            vector=embedding_vector.tolist(), 
            top_k=top_k, 
            include_metadata=True
        )
        return [match['metadata'] for match in results['matches']]

    def generate_report(self, diagnosis, confidence, similar_cases):
        """
        Generates a doctor-friendly report using LLM.
        """
        # Context from similar cases
        case_summary = "\n".join([f"- Patient {c['id']}: {c['outcome']}" for c in similar_cases])
        
        prompt = f"""
        You are an AI Medical Assistant. Analyze the following diagnosis data.
        
        Current Patient Diagnosis: {diagnosis}
        Model Confidence: {confidence:.2f}%
        
        Similar Historical Cases Retrieved:
        {case_summary}
        
        Task:
        1. Explain the diagnosis based on the confidence level.
        2. Reference the similar cases to support the conclusion.
        3. Recommend standard next steps (e.g., UPDRS assessment).
        
        Report:
        """
        return self.llm(prompt)