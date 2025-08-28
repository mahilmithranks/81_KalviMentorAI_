# main.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

"""
====================================================
   KalviMentor_AI - Zero Shot Prompting Example
====================================================

Zero-Shot Prompting:
--------------------
- Zero-shot prompting means asking the AI a question 
  WITHOUT providing any examples. 
- The model relies only on its pre-trained knowledge 
  and the prompt instructions.
- Example here: We directly ask the AI to explain 
  Newton’s Second Law of Motion in simple terms.

====================================================
"""

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Zero-Shot Prompt
zero_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
A student asks: "Explain Newton’s Second Law of Motion in simple terms."
Provide a clear, structured, and student-friendly answer.
"""

# Generate Response
response = model.generate_content(zero_shot_prompt)

# Print the Output
print("🔹 Zero-Shot Prompt:\n")
print(zero_shot_prompt.strip())
print("\n🔹 AI Response:\n")
print(response.text)
