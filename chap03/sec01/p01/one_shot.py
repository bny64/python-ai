from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

# API 키 설정 (환경 변수 'GEMINI_API_KEY'를 사용하거나 직접 문자열을 입력하세요)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="참새")]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text="짹짹")]
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="호랑이")]
        ),
    ],
    config=types.GenerateContentConfig(
        system_instruction="너는 유치원생이야. 유치원생처럼 답변해 줘",
        temperature=0.9,
    ),
)

print(response.text)
