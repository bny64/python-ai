import sys
import os
from pathlib import Path

# 루트 디렉터리를 sys.path에 추가 (util 모듈 임포트용)
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from util.gemini import api_key

from langchain_core.tools import tool
from datetime import datetime
import pytz

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)


# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f"{timezone} ({location}) 현재시각 {now}"
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str) -> str:
    """웹 검색을 수행하는 함수.
    Args:
        query (str): 검색어
        search_period (str): 검색 기간(e.g., 'w' for past week, 'm' for past month, 'y' for past year)
    Returns:
        str: 웹 검색 결과
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-kr", time=search_period)

    print("---------- WEB SEARCH ----------")
    print(query)
    print(search_period)
    print("--------------------------------")

    search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        # source='news',
        results_separator=";\n",
    )

    docs = search.invoke(query)
    return docs


# 도구 바인딩
tools = [get_current_time, get_web_search]
tool_dict = {"get_current_time": get_current_time, "get_web_search": get_web_search}

llm_with_tools = llm.bind_tools(tools)


def get_text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(get_text_content(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(
        messages
    )  # ① llm.stream()을 llm_with_tools.stream()로 변경

    gathered = None  # ②
    for chunk in response:
        text_chunk = get_text_content(chunk.content)
        if text_chunk:
            yield text_chunk

        if gathered is None:  #  ③
            gathered = chunk
        else:
            gathered += chunk

    if gathered and gathered.tool_calls:
        gathered.content = get_text_content(gathered.content)
        st.session_state.messages.append(gathered)

        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict[tool_call["name"]]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg, type(tool_msg))
            st.session_state.messages.append(tool_msg)
            # 실행된 도구의 결과를 화면(assistant/tool 영역)에 즉시 표시
            with st.chat_message("tool"):
                st.write(get_text_content(tool_msg.content))

        # 도구 실행 결과가 포함된 상태로 다시 LLM 응답을 받아와 스트리밍
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk


# Streamlit 앱
st.title("💬 Gemini Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다. "),
        AIMessage("How can I help you?"),
    ]


# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    text = get_text_content(msg.content)
    if text.strip():
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(text)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(text)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(text)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(text)


# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)  # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt))  # 사용자 메시지 저장

    with st.chat_message("assistant"):
        response = get_ai_response(st.session_state["messages"])
        result = st.write_stream(response)  # AI 메시지 출력
    
    # 마지막으로 렌더링된 최종 텍스트가 있다면 session_state에 추가 (중복 방지 체크)
    final_text = get_text_content(result)
    if final_text.strip() and not isinstance(st.session_state["messages"][-1], AIMessage):
        st.session_state["messages"].append(AIMessage(final_text))

