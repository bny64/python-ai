# from openai import OpenAI  # 주석처리
# from dotenv import load_dotenv
# import os
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
# client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

# llm = ChatOpenAI(model="gpt-4o")  # ChatOpenAI 클래스의 인스턴스 생성
llm = ChatOllama(model="deepseek-r1")


# def get_ai_response(messages):
#     response = client.chat.completions.create(
#         model="gpt-4o",  # 응답 생성에 사용할 모델 지정
#         temperature=0.9,  # 응답 생성에 사용할 temperature 설정
#         messages=messages,  # 대화 기록을 입력으로 전달
#     )
#     return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
    # {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
    SystemMessage("너는 사용자의 질문에 한국어로 답변해야."),  # 초기 시스템 메시지
]

while True:
    user_input = input("사용자: ")  # 사용자 입력 받기

    if user_input == "exit":  # ② 사용자가 대화를 종료하려는지 확인인
        break

    messages.append(
        # {"role": "user", "content": user_input} # 주석처리
        HumanMessage(user_input)
    )  # 사용자 메시지를 대화 기록에 추가

    # ai_response = get_ai_response(messages)  # 주석처리
    ai_response = llm.stream(messages)  # 대화 기록을 기반으로 AI 응답 가져오기

    ai_message = None
    for chunk in ai_response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print("")

    if "</think>" in ai_message.content:
        message_only = ai_message.content.split("</think>")[1].strip()
    else:
        message_only = ai_message.content.strip()

    messages.append(
        AIMessage(content=message_only)
    )  # AI 응답 대화 기록에 추가하기

