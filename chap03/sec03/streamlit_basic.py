import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# dotenv로 API 키 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Client 초기화
client = genai.Client(api_key=api_key)

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요? 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("메시지를 입력하세요..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Streamlit 메시지 형식을 Gemini 형식(user/model)으로 변환
        contents = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
        # Streaming API 호출
        response_stream = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=contents
        )
        
        full_response = ""
        # 스트리밍 결과 수신 및 화면 업데이트
        for chunk in response_stream:
            full_response += chunk.text
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

