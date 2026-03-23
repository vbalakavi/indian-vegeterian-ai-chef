from pathlib import Path

from config import OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "db"

class ChefAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.db = FAISS.load_local(
            str(DB_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY)

    def get_context(self, query):
        docs = self.db.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])

    def ask(self, query):
        context = self.get_context(query)

        prompt = f"""
        You are an expert Indian vegetarian chef.

        Use the context below to answer:
        {context}

        User question: {query}

        Instructions:
        - Give step-by-step recipe if asked
        - Suggest alternatives
        - Be friendly and practical
        - Keep answers clear
        """

        response = self.llm.invoke(prompt)
        return response.content
