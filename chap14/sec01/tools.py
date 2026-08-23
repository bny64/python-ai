import fitz  # PyMuPDF (PDF 텍스트 추출용)
import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from ddgs import DDGS
from langchain_core.documents import Document
from langchain_core.tools import tool

absolute_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로
current_path = os.path.dirname(absolute_path)  # 현재 .py 파일이 있는 폴더 경로


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

    resource_json_path = f'{current_path}/data/resource_{datetime.now().strftime("%Y_%m%d_%H%M%S")}.json'
    with open(resource_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results, resource_json_path
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


if __name__ == "__main__":
    result, resource_json_path = web_search.invoke("2026년 한국 경제 전망")
    print(result[0])
