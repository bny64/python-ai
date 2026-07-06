from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

# API 키 설정 (환경 변수 'GEMINI_API_KEY'를 사용하거나 직접 문자열을 입력하세요)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# 1. 채팅 세션 생성 (모델 설정 및 system_instruction 정의)
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="너는 사용자를 도와주는 상담사야.",
        temperature=0.9,
    )
)

while True:
    user_input = input("사용자: ")

    if user_input == "종료":
        break

    # 2. send_message를 호출하면 내부적으로 대화 내역이 자동으로 누적/관리됩니다.
    response = chat.send_message(user_input)

    print("상담사: ", response.text)
