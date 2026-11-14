import os
from dotenv import load_dotenv
from brain import ask_gemini

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

answer = ask_gemini(
    question="Explain Bernoulli's theorem in simple words.",
    system_prompt="You are a Mechanical Engineering tutor.",
    api_key=api_key
)

print("\nMECHMATE AI RESPONSE:\n")
print(answer)
