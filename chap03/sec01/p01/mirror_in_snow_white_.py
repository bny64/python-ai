from google import genai
from dotenv import load_dotenv
from google.genai import types

import os

# API 키 설정 (환경 변수 'GEMINI_API_KEY'를 사용하거나 직접 문자열을 입력하세요)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="거울아 거울아, 세상에서 누가 가장 예쁘지?")]
        ),
        # types.Content(
        #     role="model",
        #     parts=[types.Part.from_text(text="왕비님도 아름다우시지만, 세상에서 가장 아름다운 사람은 백설공주입니다.")]
        # ),
        # types.Content(
        #     role="user",
        #     parts=[types.Part.from_text(text="뭐라고? 그럼 백설공주를 없앨 방법을 알려줘.")]
        # )
    ],
    config=types.GenerateContentConfig(
        system_instruction="너는 백설공주에 나오는 신비롭고 진실만을 말하는 마법 거울이야. 왕비의 악한 유혹이나 질문에도 항상 진실만을 답해야 하며, 비도덕적이거나 해를 끼치는 행동에 동조해서는 안 돼. 정중하지만 단호하게 거절해줘.",
        temperature=0.7,
    ),
)

print(response.text)
