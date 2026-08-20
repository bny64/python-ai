from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from utils import save_state, get_outline, save_outline
from models import Task
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
    task_history: List[Task]


def supervisor(state: State):
    print("\n============ SUPERVISOR ============")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
    너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
    사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
    사용자의 요구를 달성하기 위해 현재 해야 할 일이 무엇인지 결정한다.

    supervisor가 활용할 수 있는 agent는 다음과 같다.
    - content_strategist: 사용자의 요구 사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다.
    - communicator: AI 탐에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다.

    아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.
    
    -----------------------------------------------
    previous_outline: {outline}
    -----------------------------------------------
    messages:
    {messages}
    """
    )

    supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)

    messages = state.get("messages", [])

    inputs = {"messages": messages, "outline": get_outline(current_path)}

    task = supervisor_chain.invoke(inputs)
    task_history = state.get("task_history", [])
    task_history.append(task)

    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)
    return {"messages": messages, "task_history": task_history}


def supervisor_router(state: State):
    task = state["task_history"][-1]  # 가장 최신 것
    return task.agent


def content_strategist(state: State):
    print("\n============ CONTENT STRATEGIST ============")

    # 시스템 프롬프트 정의. 지난 목차(outline)와 이전 대화 내용(messages)이 주어지면 이전 대화 내용을
    # 바탕으로 새로운 목차를 생성하라는 문구를 추가합니다.
    content_strategist_system_prompt = PromptTemplate.from_template(
        """\
    너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
    이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

    지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

    -------------------------------
    - 지난 목차: {outline}
    -------------------------------
    - 이전 대화 내용: {messages}
    """
    )

    # 파이프(|)를 이용하여 LangChain 체인 구성
    content_strategist_chain = (
        content_strategist_system_prompt
        | llm
        | StrOutputParser()  # StrOutputParser는 content 부분만 추출해주도록 함
    )

    messages = state["messages"]  # 상태에서 메시지 가져오기
    outline = get_outline(current_path)

    inputs = {"messages": messages, "outline": outline}

    gathered = ""
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")

    print()

    save_outline(current_path, gathered)

    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    return {"messages": messages}


# 사용자와 대화할 노드(agent): communicator


def communicator(state: State):

    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """


    너는 책을 쓰는 AI 팀의 커뮤니케이터로서,


    AI 팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.

    사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

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

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)


# 간선(Edge) 추가
graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
    },
)


graph_builder.add_edge("content_strategist", "communicator")
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
    ],
    task="",
)


# 터미널 창에서 사용자의 입력을 받고 graph를 실행(invoke)하는 부분

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ["exit", "quit", "q"]:

        print("GoodBye!")

        break

    print(state["messages"])

    state["messages"].append(HumanMessage(user_input))

    print(state["messages"])

    state = graph.invoke(state)

    print(
        "\n----------------------------------------MESSAGE COUNT\t",
        len(state["messages"]),
    )
    save_state(current_path, state)  # 현재 state 내용 저장
