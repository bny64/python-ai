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
from tools import retrieve, web_search, add_web_pages_json_to_chroma

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
    references: dict
    user_request: str  # 사용자의 요구 사항을 저장하는 변수


def business_analyst(state: State):
    print("\n============ BUSINESS ANALYST ============")

    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 비즈니스 애널리스트로서,
        AI팀의 진행상황과 "사용자 요구 사항"을 토대로,
        현 시점에서 '지난 요구 사항(previous_user_request)'과 최근 사용자의 발언을
        바탕으로 요구사항이 무엇인지 파악한다.

        다음과 같은 템플릿 형태로 반환한다.
        ```
        - 목표: 0000 \b 방법: 0000
        ```

        -----------------------------------
        *지난 요구 사항(previous_user_request)* : {previous_user_request}
        -----------------------------------
        사용자 최근 발언: {user_last_comment}
        -----------------------------------
        참고자료: {reference}
        -----------------------------------
        목차 (outline): {outline}
        -----------------------------------
        메시지 히스토리: {messages}
        -----------------------------------
        """
    )

    be_chain = business_analyst_system_prompt | llm | StrOutputParser()

    # 상태에서 메시지 가져오기
    messages = state["messages"]

    # 사용자의 마지막 발언 가져오기
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # 입력 자료 준비
    inputs = {
        "previous_user_request": state.get("user_request", None),
        "reference": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(current_path),
        "messages": messages,
        "user_last_comment": user_last_comment,
    }

    user_request = be_chain.invoke(inputs)

    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    return {"messages": messages, "user_request": user_request}


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
    - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references)에 필요한 정보가 부족한 경우 사용한다. 웹 검색을 통해 해당 정보를 벡터 DB에 보강한다.
    - vector_search_agent: 목차 작성에 필요한 자료를 확보하기 위해 벡터 DB 검색을 한다.

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


def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "vector_search_agent":
        raise ValueError(
            f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}"
        )

    vector_search_system_prompt = PromptTemplate.from_template(
        """
    너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
    목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

    현재 목차(outline)를 작성하는 데 필요한 정보를 확보하기 위해,
    다음 내용을 활용해 적절한 벡터 검색을 수행하라.

    - 검색 목적: {mission}
    -------------------------------
    - 과거 검색 내용: {reference}
    -------------------------------
    - 이전 대화 내용: {messages}
    -------------------------------
    - 목차(outline): {outline}
    """
    )

    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": mission,
        "reference": references,
        "messages": messages,
        "outline": outline,
    }

    llm_with_retriever = llm.bind_tools([retrieve])
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    search_plans = vector_search_chain.invoke(inputs)

    for tool_call in search_plans.tool_calls:
        print("------------------------------------", tool_call)

    args = tool_call["args"]

    query = args["query"]
    retrieved_docs = retrieve.invoke(args)

    references["queries"].append(query)
    references["docs"] += retrieved_docs

    unique_docs = []
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)

    references["docs"] = unique_docs

    print("Queries:-----------------------")
    queries = references["queries"]
    for query in queries:
        print(query)

    print("References:-----------------------")
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print("---------------------------")

    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다.",
        done_at="",
    )

    tasks.append(new_task)

    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)

    return {"messages": messages, "task_history": tasks, "references": references}


def content_strategist(state: State):
    print("\n============ CONTENT STRATEGIST ============")

    task_history = state.get("task_history", [])
    task = task_history[-1]

    if task.agent != "content_strategist":
        raise ValueError(
            f"Content Strategist Agent가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}"
        )
    # 시스템 프롬프트 정의. 지난 목차(outline)와 이전 대화 내용(messages)이 주어지면 이전 대화 내용을
    # 바탕으로 새로운 목차를 생성하라는 문구를 추가합니다.
    content_strategist_system_prompt = PromptTemplate.from_template(
        """\
    너는 책을 쓰는 AI 팀의 콘텐츠 전략가(Content Strategist)로서,
    이전 대화 내용을 바탕으로 사용자의 요구 사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

    지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
    목차를 작성하는 데 필요한 정보는 "참고 자료"에 있으므로 활용한다.

    다음 정보를 활용하여 목차를 작성하라.
    - 사용자 요구사항(user_request)
    - 작업(task)
    - 검색 자료(references)

    - 기존 목차(previous_outline)
    - 이전 대화 내용(messages)

    너의 작업 목표는 다음과 같다:
    1. 만약 "기존 목차 구조(previous_outline)"이 존재한다면, 사용자의 요구 사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
    - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야 한다.
    2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chpater와 section의 제목을 정한다.
    3. 책의 전반적인 세부 구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
    4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지 확인한다.
    5. 참고 자료(references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
    6. 참고 문헌은 반드시 참고 자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
    7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

    사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다.

    --------------------------------------------------
    - 사용자 요구사항(user_request):
    {user_request}
    --------------------------------------------------
    - 작업(task):
    {task}
    --------------------------------------------------
    - 참고 자료(references)
    {references}
    --------------------------------------------------
    - 기존 목차(previous_outline)
    {outline}
    --------------------------------------------------
    - 이전 대화 내용(messages)
    {messages}
    --------------------------------------------------

    작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 개수는 필요한 만큼 추가하라.
    섹션 개수는 최소 2개 이상이어야 하며, 더 많으면 좋다.

    outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
    outline_template:
    {outline_template}

    -------------------------------
    - 지난 목차: {outline}
    -------------------------------
    - 이전 대화 내용: {messages}
    -------------------------------
    - 참고 자료: {references}
    """
    )

    # 파이프(|)를 이용하여 LangChain 체인 구성
    content_strategist_chain = (
        content_strategist_system_prompt
        | llm
        | StrOutputParser()  # StrOutputParser는 content 부분만 추출해주도록 함
    )

    user_request = state.get("user_request", "")
    messages = state["messages"]  # 상태에서 메시지 가져오기
    outline = get_outline(current_path)  # 저장된 목차 가져오기

    # 템플릿 이용하기
    with open(
        f"{current_path}/templates/outline_template.md", "r", encoding="utf-8"
    ) as f:
        outline_template = f.read()

    inputs = {
        "user_request": user_request,
        "task": task,
        "messages": messages,
        "outline": outline,
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline_template": outline_template,
    }

    # 목차 작성
    gathered = ""
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end="")

    print()

    save_outline(current_path, gathered)

    if ":--- DONE ---:" in gathered:
        review = gathered.split(":--- DONE ---:")[1]
    else:
        review = gathered[-200:]

    content_strategist_message = (
        f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    )
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    # task_history = state.get("task_history", [])
    # if task_history[-1].agent != "content_strategist":
    #     raise ValueError(
    #         f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}"
    #     )

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.",
    #     done_at="",
    # )
    # task_history.append(new_task)

    return {"messages": messages, "task_history": task_history}


def web_search_agent(state: State):
    print("\n\n============ WEB SEARCH AGENT ============")

    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(
            f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}"
        )

    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로,
        목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        -------------------------------------
        - 과거 검색 내용: {references}
        -------------------------------------
        - 이전 대화 내용: {messages}
        -------------------------------------
        - 목차(outline): {outline}
        -------------------------------------
        - 현재 시각 : {current_time}
        """
    )

    messages = state.get("messages", [])
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    llm_with_web_search = llm.bind_tools([web_search])
    web_search_chain = web_search_system_prompt | llm_with_web_search

    search_plans = web_search_chain.invoke(inputs)

    queries = []

    for tool_call in search_plans.tool_calls:
        print("-------- web search --------", tool_call)
        args = tool_call["args"]

        queries.append(args["query"])
        # 검색 결과 JSON 파일 경로 가져오기
        _, json_path = web_search.invoke(args)

        # JSON 파일을 크로마 DB에 추가
        add_web_pages_json_to_chroma(json_path)

    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_desc = (
        "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    )
    task_desc += f"다음 항목이 새로 추가되었다.\n: {queries}"

    new_task = Task(
        agent="vector_search_agent", done=False, description=task_desc, done_at=""
    )

    tasks.append(new_task)

    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    return {"messages": messages, "task_history": tasks}


# 사용자와 대화할 노드(agent): communicator


def communicator(state: State):

    print("\n\n============ COMMUNICATOR ============")

    communicator_system_prompt = PromptTemplate.from_template(
        """


    너는 책을 쓰는 AI 팀의 커뮤니케이터로서,


    AI 팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.

    사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

    outline: {outline}
    -----------------------------------------------
    messages: {messages}
    -----------------------------------------------
    """
    )

    system_chain = communicator_system_prompt | llm

    # 상태 메시지 가져오기

    messages = state["messages"]

    # 입력값 정의

    inputs = {"messages": messages, "outline": get_outline(current_path)}

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

    task_history = state.get("task_history", [])
    if task_history[-1].agent != "communicator":
        raise ValueError(
            f"Content Strategist가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}"
        )

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {"messages": messages, "task_history": task_history}


# 상태 그래프 정의


graph_builder = StateGraph(State)


# 노드 추가

graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)


# 간선(Edge) 추가
graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
        "vector_search_agent": "vector_search_agent",
        "web_search_agent": "web_search_agent",
    },
)

graph_builder.add_edge("content_strategist", "business_analyst")
graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "business_analyst")
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
    task_history=[],
    references={"queries": [], "docs": []},
    user_request="",
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
