from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 임베딩 모델 선언
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", google_api_key=GEMINI_API_KEY
)

# 언어 모델 불러오기
from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=GEMINI_API_KEY)

# Load Chroma store
from langchain_chroma import Chroma

print("Loading existing Chroma store")
persist_directory = "D:/workspace/python-ai/chap09/chroma_store"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Create retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser  # 문자열 출력 파서 불러오기

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = (
    create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()
)

# query agumentation chain
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한 문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라:\n\n{query}",
        ),
    ]
)


query_augmentation_chain = (
    query_augmentation_prompt | llm | StrOutputParser()
)
