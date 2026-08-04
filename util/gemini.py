import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# dotenv로 API 키 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")