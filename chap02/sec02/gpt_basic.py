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
    contents="2022년 월드컵 우승 팀은 어디지? 히스토리까지.",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.1,
    ),
)

print(response)
print("-----")
print(response.text)
