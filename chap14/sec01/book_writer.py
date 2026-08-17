from langgraph.graph import StateGraph, END, START


from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage


from langchain_core.prompts import PromptTemplate


from typing_extensions import TypedDict


from typing import List


from utils import save_state


from datetime import datetime
import os


import dotenv


# 현재 폴더 경로 찾기


# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 사용


filename = os.path.basename(__file__)  # 현재 파일명 반환


absolute_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로 반환


current_path = os.path.dirname(absolute_path)  # 현재 .py 파일이 있는 폴더 경로


dotenv.load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# 모델 초기화


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", google_api_key=os.getenv("GEMINI_API_KEY")
)


# 상태 정의


class State(TypedDict):

    messages: List[AnyMessage | str]


# 사용자와 대화할 노드(agent): communicator


def communicator(state: State):

    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """


    너는 책을 쓰는 AI 팀의 커뮤니케이터로서,


    AI 팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.



    messages: {messages}
    """
    )

    system_chain = communicator_system_prompt | llm

    # 상태 메시지 가져오기

    messages = state["messages"]

    # 입력값 정의

    inputs = {"messages": messages}

    gathered = None

    print("\nAI\t:", end="")

    for chunk in system_chain.stream(inputs):
        content = chunk.content
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content)
        print(text, end="", flush=True)

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    return {"messages": messages}


# 상태 그래프 정의


graph_builder = StateGraph(State)


# 노드 추가


graph_builder.add_node("communicator", communicator)


# 간선(Edge) 추가


graph_builder.add_edge(START, "communicator")


graph_builder.add_edge("communicator", END)


# 그래프 컴파일


graph = graph_builder.compile()


# 그래프 도식화


graph.get_graph().draw_mermaid_png(
    output_file_path=absolute_path.replace(".py", "_graph.png")
)


print(absolute_path.replace(".py", ".png"))


# 상태 초기화


state = State(
    messages=[
        SystemMessage(
            f"""


        너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.


        사용자가 사용하는 언어로 대화하라.



        현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.
        """
        )
    ]
)


# 터미널 창에서 사용자의 입력을 받고 graph를 실행(invoke)하는 부분

# while True:

#     user_input = input("\nUser\t: ").strip()

#     if user_input.lower() in ["exit", "quit", "q"]:

#         print("GoodBye!")

#         break

#     state["messages"].append(HumanMessage(user_input))

#     state = graph.invoke(state)


# 터미널 창에서 사용자의 입력을 받고 graph를 실행(invoke)하는 부분

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ["exit", "quit", "q"]:

        print("GoodBye!")

        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print(
        "\n----------------------------------------MESSAGE COUNT\t",
        len(state["messages"]),
    )
    save_state(current_path, state)  # 현재 state 내용 저장
