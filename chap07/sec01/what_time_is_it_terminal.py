from gpt_functions import get_current_time

from google import genai
from dotenv import load_dotenv
from google.genai import types

import os
import io
import urllib.request
from PIL import Image

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def get_ai_response(messages, tools=None):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction='너는 사용자를 도와주는 상담사야.',
            tools=tools
        ),
    )

    return response

# tools 매핑에 실제 파이썬 함수 객체 전달
tools_list = [get_current_time]

while True:
    user_input = input('사용자\t: ')
    response = get_ai_response(user_input, tools=tools_list)
    print('AI\t:', response.text)