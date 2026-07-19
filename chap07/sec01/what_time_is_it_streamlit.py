from gpt_functions import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations

from google import genai
from dotenv import load_dotenv
from google.genai import types

import os
import streamlit as st

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="시간 확인 챗봇", page_icon="⏰")
st.title("⏰ 시간 확인 챗봇")

# 1. client 초기화 및 st.session_state에 저장
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=api_key)

# 2. 대화 세션(chat) 초기화 및 st.session_state에 저장
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="너는 사용자를 도와주는 상담사야.",
            tools=[get_current_time, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations],
        ),
    )

# 3. 기존 대화 기록을 화면에 표시
# (함수 호출/결과값 전송 같은 내부 메시지는 제외하고 일반 텍스트 대화만 표시)
for message in st.session_state.chat.get_history():
    text_content = ""
    if message.parts:
        for part in message.parts:
            if part.text:
                text_content += part.text

    if text_content:
        role = "user" if message.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(text_content)

# 4. 사용자 입력 받기
if user_input := st.chat_input("메시지를 입력하세요 (예: 뉴욕 지금 몇 시야?)"):
    # 사용자 입력 화면 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 답변 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = st.session_state.chat.send_message(user_input)

                # 함수 호출 디버깅 로그 출력 (화면 캡션 형식)
                history = st.session_state.chat.get_history()
                # 최근 추가된 메시지들 중에서 model의 function_call을 찾아 출력합니다.
                for message in reversed(history):
                    # 사용자의 실제 텍스트 질문(function_response가 아닌 일반 텍스트 입력)을 만나면 탐색을 중단합니다.
                    if message.role == "user":
                        has_text = False
                        if message.parts:
                            for part in message.parts:
                                if part.text:
                                    has_text = True
                                    break
                        if has_text:
                            break
                    
                    if message.role == "model" and message.parts:
                        for part in message.parts:
                            if part.function_call:
                                st.caption(
                                    f"🔧 **실행된 도구**: `{part.function_call.name}` (인자: `{part.function_call.args}`)"
                                )

                # 최종 답변 텍스트 출력
                st.markdown(response.text)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
