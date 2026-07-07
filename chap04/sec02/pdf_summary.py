from google import genai
from dotenv import load_dotenv
from google.genai import types

import os
import pymupdf

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def pdf_to_text(pdf_file_path: str):
    doc = pymupdf.open(pdf_file_path)

    header_height = 80
    footer_height = 80

    full_text = ""

    for page in doc:

        rect = page.rect

        header = page.get_text(clip=(0, 0, rect.width, header_height))
        footer = page.get_text(
            clip=(0, rect.height - footer_height, rect.width, rect.height)
        )
        text = page.get_text(
            clip=(0, header_height, rect.width, rect.height - footer_height)
        )

        full_text += text + "\n--------------------------------\n"

    # 파일명만 추출

    pdf_file_name = os.path.basename(pdf_file_path)
    pdf_file_name = os.path.splitext(pdf_file_name)[0]  # 확장자 제거

    txt_file_path = f"chap04/output/{pdf_file_name}_with_preprocessing.txt"
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return txt_file_path


def summarize_text(file_path=str):

    client = genai.Client(api_key=api_key)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    system_prompt = f"""
        너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라.

        작성해야 하는 포맷은 다음과 같다.

        # 제목

        ## 제자의 문제 인식 및 주장(15문장 이내)
        
        ## 저자 소개

        ===== 이하 텍스트 ===

        { text }
    """

    print(system_prompt)
    print("==========")

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text=system_prompt)]
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )

    return response.text


def summarize_pdf(pdf_file_path: str, output_file_path: str):
    txt_file_path = pdf_to_text(pdf_file_path)
    summary = summarize_text(txt_file_path)

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    pdf_file_path = "chap04/data/KCI_FI003103066.pdf"
    summarize_pdf(pdf_file_path, "chap04/output/crop_model_summary2.txt")
