from google import genai
from dotenv import load_dotenv
from google.genai import types

import os
import streamlit as st

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Gemini 스트리밍 챗봇", page_icon="💬")
st.title("💬 Gemini 스트리밍 챗봇")

# 1. client 초기화 및 st.session_state에 저장
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=api_key)

# 2. 대화 세션(chat) 초기화 및 st.session_state에 저장 (도구(tools) 없이 최소 구성)
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="너는 친절하고 유용한 인공지능 어시스턴트야.",
        ),
    )

# 3. 기존 대화 기록을 화면에 표시 (스트리밍 출력을 제외하고 단순 텍스트 표시)
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
if user_input := st.chat_input("메시지를 입력하세요 (예: 오늘 날씨 어때?)"):
    # 사용자 입력 화면 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 스트리밍 답변 생성 및 표시
    with st.chat_message("assistant"):
        # generator 함수로 스트리밍 데이터를 yield하여 st.write_stream에 넘겨줍니다.
        def stream_generator():
            response_stream = st.session_state.chat.send_message_stream(user_input)
            for i, chunk in enumerate(response_stream):
                if chunk.text:
                    # chunk 수신 확인용 터미널 로그 출력
                    print(f"[Chunk #{i}] {repr(chunk.text)}")
                    yield chunk.text

        st.write_stream(stream_generator())
