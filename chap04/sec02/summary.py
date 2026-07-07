from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

# API 키 설정 (환경 변수 'GEMINI_API_KEY'를 사용하거나 직접 문자열을 입력하세요)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


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
        model="gemini-1.5-flash",
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


if __name__ == "__main__":
    file_path = "./chap04/output/KCI_FI003103066_with_preprocessing.txt"

    summary = summarize_text(file_path)

    with open("./chap04/output/crop_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
