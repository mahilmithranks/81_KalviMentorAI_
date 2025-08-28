# main.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

"""
====================================================
   KalviMentor_AI - Prompt Engineering & Evaluation
====================================================

This file demonstrates:
1. Zero-Shot Prompting
2. One-Shot Prompting
3. Multi-Shot Prompting
4. Dynamic Prompting
5. Chain-of-Thought Prompting
6. Evaluation Pipeline with Judge Prompt
====================================================
"""

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")


# ==================================================
# 1. Zero-Shot Prompt
# ==================================================
zero_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
A student asks: "Explain Newton’s Second Law of Motion in simple terms."
Provide a clear, structured, and student-friendly answer.
"""

zero_shot_response = model.generate_content(zero_shot_prompt)


# ==================================================
# 2. One-Shot Prompt
# ==================================================
one_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
Example:
Q: What is Photosynthesis?
A: Photosynthesis is how plants make food using sunlight, water, and carbon dioxide.
They turn these into glucose (sugar) for energy and release oxygen.

Now, answer the student’s question:
Q: Explain the process of Evaporation.
"""

one_shot_response = model.generate_content(one_shot_prompt)


# ==================================================
# 3. Multi-Shot Prompt
# ==================================================
multi_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
Here are examples:

Q: What is gravity?
A: Gravity is the force that pulls objects toward each other, like how Earth pulls us down.

Q: What is friction?
A: Friction is a force that happens when two surfaces rub against each other, slowing things down.

Now, answer the student’s question:
Q: Explain Electricity in simple terms.
"""

multi_shot_response = model.generate_content(multi_shot_prompt)


# ==================================================
# 4. Dynamic Prompting
# ==================================================
student_level = "beginner"  # can be beginner, intermediate, advanced
student_topic = "Black Holes"

dynamic_prompt = f"""
You are KalviMentor_AI, a friendly educational mentor.
The student is at a {student_level} level.
Explain the topic: {student_topic}.
Keep the explanation tailored to their level.
"""

dynamic_response = model.generate_content(dynamic_prompt)


# ==================================================
# 5. Chain-of-Thought Prompting
# ==================================================
chain_of_thought_prompt = """
You are KalviMentor_AI, a friendly mentor.
A student asks: "Solve 12 + 28 × 2"

Think step by step:
1. Recall order of operations (BODMAS).
2. First, multiply 28 × 2.
3. Then, add 12 to the result.
4. Show reasoning clearly and provide the final answer.
"""

chain_of_thought_response = model.generate_content(chain_of_thought_prompt)


# ==================================================
# 6. Evaluation Pipeline
# ==================================================

# --- Dataset (5 samples) ---
dataset = [
    {"query": "What is gravity?", "expected_answer": "Gravity is the force that pulls objects toward each other."},
    {"query": "What is evaporation?", "expected_answer": "Evaporation is when liquid water changes into water vapor."},
    {"query": "What is photosynthesis?", "expected_answer": "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide."},
    {"query": "Solve 5 + 3 × 2", "expected_answer": "11"},
    {"query": "What is friction?", "expected_answer": "Friction is the force that slows motion when two surfaces rub together."}
]

# --- Judge Prompt Template ---
def judge_response(query, expected, actual):
    judge_prompt = f"""
You are KalviMentor_AI Judge.
Evaluate the model’s response compared to the expected answer.

Student Query: {query}
Expected Answer: {expected}
Model Answer: {actual}

Judge the answer on:
1. Correctness (Is it factually correct?)
2. Clarity (Is it understandable for a student?)
3. Relevance (Does it answer the query?)

Give a short evaluation and a score between 1 (poor) to 5 (excellent).
"""
    return model.generate_content(judge_prompt).text

# --- Run Evaluation ---
def run_evaluation():
    print("\n================= Evaluation Pipeline =================\n")
    for i, sample in enumerate(dataset, 1):
        # Get model response
        query = sample["query"]
        expected = sample["expected_answer"]
        actual_response = model.generate_content(query).text

        # Judge the response
        evaluation = judge_response(query, expected, actual_response)

        # Print results
        print(f"Test Case {i}")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Model: {actual_response}")
        print(f"Evaluation: {evaluation}")
        print("-------------------------------------------------------")


# ==================================================
# Run All Sections
# ==================================================
if __name__ == "__main__":
    print("🔹 Zero-Shot Response:\n", zero_shot_response.text, "\n")
    print("🔹 One-Shot Response:\n", one_shot_response.text, "\n")
    print("🔹 Multi-Shot Response:\n", multi_shot_response.text, "\n")
    print("🔹 Dynamic Response:\n", dynamic_response.text, "\n")
    print("🔹 Chain-of-Thought Response:\n", chain_of_thought_response.text, "\n")

    # Run Evaluation
    run_evaluation()
