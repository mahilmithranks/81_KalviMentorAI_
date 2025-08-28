# main.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

"""
====================================================
   KalviMentor_AI - One Shot Prompting Example
====================================================

One-Shot Prompting:
-------------------
- One-shot prompting means giving the AI a single example 
  before asking it to answer a new question. 
- This example guides the model on style, tone, and structure.
- Example here: We first show how KalviMentor_AI explains 
  "Photosynthesis" and then ask it to explain 
  "Cellular Respiration."

====================================================
"""

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# One-Shot Prompt (Example + New Question)
one_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.

Example:
Q: Explain Photosynthesis in simple terms.
A:
1. Concept Explanation: Photosynthesis is the process by which green plants make their own food using sunlight, water, and carbon dioxide.
2. Example or Analogy: Think of it like a plant’s kitchen – sunlight is the stove, water is an ingredient, and carbon dioxide is another ingredient. Together, they cook up sugar (food) for the plant.
3. Practice Question: Why is sunlight important for photosynthesis?
4. Feedback/Tips: Remember that plants also release oxygen during this process, which is essential for humans.

---

Now, the student asks:
Q: Explain Cellular Respiration in simple terms.
"""

# Generate Response
response = model.generate_content(one_shot_prompt)

# Print the Output
print("🔹 One-Shot Prompt:\n")
print(one_shot_prompt.strip())
print("\n🔹 AI Response:\n")
print(response.text)
