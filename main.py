# main.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

"""
====================================================
   KalviMentor_AI - Multi Shot Prompting Example
====================================================

Multi-Shot Prompting:
---------------------
- Multi-shot prompting means giving the AI multiple 
  examples before asking it to answer a new question. 
- This helps the AI understand the pattern, tone, and 
  format more clearly than one-shot prompting.

Example here:
1. Example Q&A about Photosynthesis
2. Example Q&A about Newton's Second Law
Then we ask about Plate Tectonics.

====================================================
"""

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Multi-Shot Prompt (Multiple Examples + New Question)
multi_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.

Example 1:
Q: Explain Photosynthesis in simple terms.
A:
1. Concept Explanation: Photosynthesis is the process by which green plants make their own food using sunlight, water, and carbon dioxide.
2. Example or Analogy: Think of it like a plant’s kitchen – sunlight is the stove, water is an ingredient, and carbon dioxide is another ingredient. Together, they cook up sugar (food) for the plant.
3. Practice Question: Why is sunlight important for photosynthesis?
4. Feedback/Tips: Remember that plants also release oxygen during this process, which is essential for humans.

---

Example 2:
Q: Explain Newton’s Second Law of Motion in simple terms.
A:
1. Concept Explanation: Newton’s Second Law says that Force = Mass × Acceleration (F = ma). It tells us how much force is needed to move something.
2. Example or Analogy: Imagine pushing a shopping cart. An empty cart (less mass) moves easily, but a full cart (more mass) needs more force to accelerate.
3. Practice Question: If you apply the same force to a light and a heavy object, which one accelerates more?
4. Feedback/Tips: Always link the concept back to real-life experiences for better understanding.

---

Now, the student asks:
Q: Explain Plate Tectonics in simple terms.
"""

# Generate Response
response = model.generate_content(multi_shot_prompt)

# Print the Output
print("🔹 Multi-Shot Prompt:\n")
print(multi_shot_prompt.strip())
print("\n🔹 AI Response:\n")
print(response.text)
