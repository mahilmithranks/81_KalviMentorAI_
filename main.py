# main.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

"""
====================================================
   KalviMentor_AI - System & User Prompt (RTFC)
====================================================

RTFC Framework:
---------------
R = Role       → Define AI's identity
T = Task       → What AI must do
F = Format     → Output structure
C = Constraints→ Rules, limits, style

Example:
We define KalviMentor_AI’s role, task, format, and constraints,
then pass a student’s query as the user prompt.

====================================================
"""

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# System Prompt (RTFC)
system_prompt = """
[Role]  
You are KalviMentor_AI, a friendly and knowledgeable educational mentor.

[Task]  
Guide students in their learning journey by explaining concepts, generating practice questions,
evaluating answers, and providing constructive feedback.

[Format]  
Always respond in a clear, structured way using:
1. Concept Explanation
2. Example or Analogy
3. Practice Question
4. Feedback / Tips

[Constraints]  
- Keep the tone encouraging and student-friendly.
- Avoid overly complex jargon.
- Use step-by-step reasoning when needed.
"""

# User Prompt (student query)
user_prompt = """
A student asks: "Can you explain the difference between mitosis and meiosis in simple terms?"
"""

# Combine prompts
full_prompt = f"{system_prompt}\n\n{user_prompt}"

# Generate Response
response = model.generate_content(full_prompt)

# Print the Output
print("🔹 System Prompt + User Prompt (RTFC):\n")
print(full_prompt.strip())
print("\n🔹 AI Response:\n")
print(response.text)
