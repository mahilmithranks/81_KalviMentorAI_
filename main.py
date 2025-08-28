# main.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

"""
====================================================
   KalviMentor_AI - Prompting Techniques Demo
====================================================

This script demonstrates multiple prompting strategies:

1. Zero-Shot Prompting
2. One-Shot Prompting
3. Multi-Shot Prompting
4. Dynamic Prompting
5. System + User Prompting (RTFC Framework)
6. Chain-of-Thought Prompting

====================================================
"""

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------------------------------
# 1. Zero-Shot Prompting
# --------------------------------------------------
zero_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
A student asks: "Explain Newton’s Second Law of Motion in simple terms."
Provide a clear, structured, and student-friendly answer.
"""

# --------------------------------------------------
# 2. One-Shot Prompting
# --------------------------------------------------
one_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.

Example:
Q: Explain Photosynthesis in simple terms.
A:
1. Concept Explanation: Photosynthesis is the process by which plants make food using sunlight, water, and carbon dioxide.
2. Example or Analogy: Think of it like a kitchen – sunlight is the stove, water and CO2 are ingredients, and sugar is the cooked food.
3. Practice Question: Why is sunlight important for photosynthesis?
4. Feedback/Tips: Don’t forget that plants release oxygen too.

---

Now, the student asks:
Q: Explain Cellular Respiration in simple terms.
"""

# --------------------------------------------------
# 3. Multi-Shot Prompting
# --------------------------------------------------
multi_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.

Example 1:
Q: Explain Photosynthesis in simple terms.
A:
1. Concept Explanation: Plants make food using sunlight, water, and CO2.
2. Example or Analogy: Like a kitchen cooking sugar.
3. Practice Question: Why is sunlight important for photosynthesis?
4. Feedback/Tips: Remember plants release oxygen.

---

Example 2:
Q: Explain Newton’s Second Law in simple terms.
A:
1. Concept Explanation: Force = Mass × Acceleration.
2. Example or Analogy: Pushing a heavy cart vs a light cart.
3. Practice Question: If the same force is applied, which accelerates more?
4. Feedback/Tips: Link it to daily life.

---

Now, the student asks:
Q: Explain Plate Tectonics in simple terms.
"""

# --------------------------------------------------
# 4. Dynamic Prompting
# --------------------------------------------------
student_level = "Beginner"   # could be Beginner, Intermediate, Advanced
topic = "Artificial Intelligence"

dynamic_prompt = f"""
You are KalviMentor_AI, a friendly educational mentor.

The student is at a **{student_level}** level.

Your task is to explain the topic: "{topic}".

Instructions:
- Adapt explanation to the student’s level.
- If Beginner → use simple words and analogies.
- If Intermediate → give real-world examples.
- If Advanced → provide in-depth details.
Always include:
1. Concept Explanation
2. Example or Analogy
3. Practice Question
4. Feedback/Tips
"""

# --------------------------------------------------
# 5. System + User Prompt (RTFC Framework)
# --------------------------------------------------
system_prompt = """
[Role]
You are KalviMentor_AI, a friendly and knowledgeable educational mentor.

[Task]
Guide students by explaining concepts, generating practice questions,
evaluating answers, and providing constructive feedback.

[Format]
Always respond in a clear, structured way:
1. Concept Explanation
2. Example or Analogy
3. Practice Question
4. Feedback/Tips

[Constraints]
- Keep tone encouraging and student-friendly.
- Avoid jargon unless necessary.
- Use step-by-step reasoning.
"""

user_prompt = """
A student asks: "Can you explain the difference between mitosis and meiosis in simple terms?"
"""

rtfc_prompt = f"{system_prompt}\n\n{user_prompt}"

# --------------------------------------------------
# 6. Chain-of-Thought Prompting
# --------------------------------------------------
cot_prompt = """
You are KalviMentor_AI, a mentor who explains reasoning step by step.

Question:
A car accelerates from rest to 20 m/s in 5 seconds. 
Its mass is 1000 kg. What force is applied?

Answer format:
1. Step-by-step reasoning: Show how to calculate acceleration, then use F = m × a.
2. Final Answer: Provide the force in Newtons.
"""

# --------------------------------------------------
# Mode Selector
# --------------------------------------------------
modes = {
    "zero_shot": zero_shot_prompt,
    "one_shot": one_shot_prompt,
    "multi_shot": multi_shot_prompt,
    "dynamic": dynamic_prompt,
    "rtfc": rtfc_prompt,
    "chain_of_thought": cot_prompt
}

# Choose the mode here (change this for testing)
selected_mode = "chain_of_thought"   # options: zero_shot, one_shot, multi_shot, dynamic, rtfc, chain_of_thought

# Run the selected prompt
print(f"\n🔹 Running Mode: {selected_mode.upper()}\n")
response = model.generate_content(modes[selected_mode])

# Print prompts & output
print("🔹 Prompt Sent:\n")
print(modes[selected_mode].strip())
print("\n🔹 AI Response:\n")
print(response.text)
