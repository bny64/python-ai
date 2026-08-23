import os
import json


def save_state(current_path, state):
    """현재 상태(메시지 내역 등)를 JSON 파일로 저장합니다.

    Args:
        current_path (str): 데이터 폴더가 위치할 현재 경로
        state (dict): 저장할 상태 정보 (messages 키에 LangChain 메시지 목록 포함)
    """
    # 저장할 디렉토리가 없으면 생성합니다.
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")

    state_dict = {}
    # 메시지 객체에서 클래스 이름과 텍스트 콘텐츠만 추출하여 튜플 목록으로 변환합니다.
    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    state_dict["task_history"] = [
        task.to_dict() for task in state.get("task_history", [])
    ]

    # references
    references = state.get("references", {"queries": [], "docs": []})
    state_dict["references"] = {
        "queries": references["queries"],
        "docs": [doc.metadata for doc in references["docs"]],
    }

    # JSON 파일로 상태를 기록합니다.
    with open(f"{current_path}/data/state.json", "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)


def get_outline(current_path):
    """저장된 목차(Markdown) 파일의 내용을 읽어옵니다.

    Args:
        current_path (str): 데이터 폴더가 위치한 현재 경로

    Returns:
        str: 목차 내용 (파일이 없으면 기본 메시지 반환)
    """
    outline = "아직 작성된 목차가 없습니다."

    # 목차 파일이 존재하면 읽어옵니다.
    if os.path.exists(f"{current_path}/data/outline.md"):
        with open(f"{current_path}/data/outline.md", "r", encoding="utf-8") as f:
            outline = f.read()
    return outline


def save_outline(current_path, outline):
    """목차(Markdown) 내용을 파일로 저장합니다.

    Args:
        current_path (str): 데이터 폴더가 위치할 현재 경로
        outline (str): 저장할 목차 텍스트 내용

    Returns:
        str: 저장된 목차 내용
    """
    # 저장할 디렉토리가 없으면 생성합니다.
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    # 파일에 목차 내용을 작성합니다.
    with open(f"{current_path}/data/outline.md", "w", encoding="utf-8") as f:
        f.write(outline)
    return outline

