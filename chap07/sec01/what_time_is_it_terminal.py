from gpt_functions import get_current_time

from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# 1. client.chats.create를 사용하여 대화 세션을 생성합니다.
# 2. tools 에 get_current_time 함수를 등록하면 google-genai SDK가
#    자동 함수 호출(Automatic Function Calling)을 처리하여 필요한 경우 실행하고 응답합니다.
chat = client.chats.create(
    model="gemini-3.1-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction="너는 사용자를 도와주는 상담사야.",
        tools=[get_current_time]
    )
)

while True:
    try:
        user_input = input("사용자\t: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("대화를 종료합니다.")
            break
        
        response = chat.send_message(user_input)
        
        # 대화 기록(history)을 역순으로 탐색하여 가장 최근에 발생한 함수 호출 정보를 찾아 출력합니다.
        history = chat.get_history()
        function_called = False
        for content in reversed(history):
            # 사용자의 질문 하나에 대해 모델이 함수 호출을 요청(role: "model")한 메시지를 찾습니다.
            if content.role == "model" and content.parts:
                for part in content.parts:
                    if part.function_call:
                        print(f"디버그 (호출된 함수): {part.function_call.name}")
                        print(f"디버그 (전달된 인자): {part.function_call.args}")
                        function_called = True
                        break
            if function_called:
                break
        
        if not function_called:
            print("디버그 (호출된 함수 없음)")

        print("AI\t:", response.text)
    except KeyboardInterrupt:
        print("\n대화를 종료합니다.")
        break
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
