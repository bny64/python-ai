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
        system_instruction="너는 배트맨에 나오는 조커야. 조커의 악당 캐릭터에 맞게 답변해 줘.",
        temperature=0.7,
    ),
)

print(response.text)
