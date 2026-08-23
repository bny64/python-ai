import fitz  # PyMuPDF (PDF 텍스트 추출용)
import json
import os
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime
from ddgs import DDGS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

absolute_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로
current_path = os.path.dirname(absolute_path)  # 현재 .py 파일이 있는 폴더 경로

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GEMINI 임베딩 모델을 사용하여 벡터 저장소(Chroma DB)를 생성
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", google_api_key=GEMINI_API_KEY
)

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)


@tool
def web_search(query: str):
    """DuckDuckGo 및 크롤러를 이용하여 웹 검색과 전체 본문(raw_content)을 수집하는 도구."""
    results = DDGS().text(query, region="kr-kr", max_results=5)

    for result in results:
        # DuckDuckGo는 기본적으로 2~3줄 요약(body)만 제공하므로
        # Tavily처럼 전체 본문을 얻으려면 load_web_page(href)를 무조건 호출해야 합니다.
        try:
            result["raw_content"] = load_web_page(result["href"])
        except Exception as e:
            print(f"Error loading page: {result.get('href')} -> {e}")
            result["raw_content"] = result.get("body", "")

    resource_json_path = (
        f'{current_path}/data/resource_{datetime.now().strftime("%Y_%m%d_%H%M%S")}.json'
    )
    with open(resource_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results, resource_json_path


def web_page_to_document(web_page):

    page_content = web_page["raw_content"]

    document = Document(
        page_content=page_content,
        metadata={
            "title": web_page["title"],
            "source": web_page["href"],
        },
    )

    return document


def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding="UTF-8") as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print("Splitting documents...")
    print(
        f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits


def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    urls = [document.metadata["source"] for document in documents]

    stored_metadatas = vectorstore._collection.get()["metadatas"]
    stored_web_urls = [metadata["source"] for metadata in stored_metadatas]

    new_urls = set(urls) - set(stored_web_urls)

    new_documents = []

    for document in documents:
        if document.metadata["source"] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    splits = split_documents(new_documents, chunk_size, chunk_overlap)

    if splits:
        batch_size = 50
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            print(f"Adding documents {i} to {min(i + batch_size, len(splits))}...")
            vectorstore.add_documents(batch)
            if i + batch_size < len(splits):
                time.sleep(60)
    else:
        print("No new urls to process")


def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(documents, chunk_size, chunk_overlap)


def load_web_page(url: str):
    """웹 페이지(HTML) 또는 PDF URL에서 텍스트를 파싱하여 반환하는 함수."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)

    content_type = response.headers.get("Content-Type", "").lower()
    is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type

    if is_pdf:
        # PDF 바이너리를 PyMuPDF(fitz)로 열어 텍스트 추출
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc if page.get_text()])
    else:
        # 일반 HTML 웹페이지 파싱
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

    raw_content = text.strip()
    while "\n\n\n" in raw_content or "\t\t\t" in raw_content:
        raw_content = raw_content.replace("\n\n\n", "\n\n")
        raw_content = raw_content.replace("\t\t\t", "\t\t")

    return raw_content


@tool
def retrieve(query: str, top_k: int = 5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs


if __name__ == "__main__":
    # documents = web_page_json_to_document(
    #     f"{current_path}/data/resource_2026_0823_150903.json"
    # )

    # splits = split_documents(documents)
    # print(splits)
    # add_web_pages_json_to_chroma(f"{current_path}/data/resource_2026_0823_150903.json")

    retrieved_docs = retrieve.invoke({"query": "한국 경제 위험 요소 "})
    print(retrieved_docs)
