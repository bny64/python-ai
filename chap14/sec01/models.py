from pydantic import BaseModel, Field
from typing import Literal


class Task(BaseModel):
    agent: Literal["content_strategist", "communicator", "vector_search_agent"] = Field(
        ...,
        description="""
  너는 책을 쓰는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
  사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고,
  사용자의 요구를 달성하기 위해 현재 해야 할 일이 무엇인지 결정한다.
  """,
    )

    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="작업 설명")
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간. ")

    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at,
        }
