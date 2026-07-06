from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

# API 키 설정 (환경 변수 'GEMINI_API_KEY'를 사용하거나 직접 문자열을 입력하세요)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

while True:
    user_input = input("사용자: ")

    if user_input == "종료":
        break

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)]),
        ],
        config=types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 상담사야.",
            temperature=0.9,
        ),
    )

    print("상담사: ", response.text)
