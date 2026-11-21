import os
import sys
from dotenv import load_dotenv
from brain import ask_gemini, ask_gemini_stream

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY is not set.")
    print("Please add your key to .env (e.g. GEMINI_API_KEY=your_key)")
    sys.exit(1)

print("Testing Gemini streaming response...\n")
question = "Explain Bernoulli's theorem in simple words with its formula."

try:
    print("--- Streaming Output ---")
    for chunk in ask_gemini_stream(
        question=question,
        system_prompt="You are a Mechanical Engineering tutor.",
        api_key=api_key
    ):
        print(chunk, end="", flush=True)
    print("\n\nTest passed successfully!")
except Exception as e:
    print(f"\nFailed to connect to Gemini API: {e}")
    sys.exit(1)
