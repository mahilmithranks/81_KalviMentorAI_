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
7. Token Logging
8. Temperature Control (creativity vs. determinism)

====================================================
"""

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini Model with Temperature
generation_config = {
    "temperature": 0.7,  # <-- adjust this value (0.0 to 1.0)
    "max_output_tokens": 500,
    "top_p": 0.9,
    "top_k": 40
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)


# ==================================================
# Utility: Generate with token logging
# ==================================================
def generate_with_logging(prompt):
    """Generate model output and log token usage."""
    response = model.generate_content(prompt)
    
    # Token usage metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print(f"🔹 Tokens Used - Prompt: {usage.prompt_token_count}, "
              f"Candidates: {usage.candidates_token_count}, "
              f"Total: {usage.total_token_count}\n")
    else:
        print("⚠️ Token usage metadata not available.\n")

    return response


# ==================================================
# Prompt Examples (Zero-shot, One-shot, Multi-shot, etc.)
# ==================================================
zero_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
A student asks: "Explain Newton’s Second Law of Motion in simple terms."
Provide a clear, structured, and student-friendly answer.
"""
zero_shot_response = generate_with_logging(zero_shot_prompt)

one_shot_prompt = """
You are KalviMentor_AI, a friendly educational mentor.
Example:
Q: What is Photosynthesis?
A: Photosynthesis is how plants make food using sunlight, water, and carbon dioxide.

Now, answer the student’s question:
Q: Explain the process of Evaporation.
"""
one_shot_response = generate_with_logging(one_shot_prompt)

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
multi_shot_response = generate_with_logging(multi_shot_prompt)

student_level = "beginner"
student_topic = "Black Holes"
dynamic_prompt = f"""
You are KalviMentor_AI, a friendly educational mentor.
The student is at a {student_level} level.
Explain the topic: {student_topic}.
Keep the explanation tailored to their level.
"""
dynamic_response = generate_with_logging(dynamic_prompt)

chain_of_thought_prompt = """
You are KalviMentor_AI, a friendly mentor.
A student asks: "Solve 12 + 28 × 2"

Think step by step:
1. Recall order of operations (BODMAS).
2. First, multiply 28 × 2.
3. Then, add 12 to the result.
4. Show reasoning clearly and provide the final answer.
"""
chain_of_thought_response = generate_with_logging(chain_of_thought_prompt)


# ==================================================
# Evaluation Pipeline
# ==================================================
dataset = [
    {"query": "What is gravity?", "expected_answer": "Gravity is the force that pulls objects toward each other."},
    {"query": "What is evaporation?", "expected_answer": "Evaporation is when liquid water changes into water vapor."},
    {"query": "What is photosynthesis?", "expected_answer": "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide."},
    {"query": "Solve 5 + 3 × 2", "expected_answer": "11"},
    {"query": "What is friction?", "expected_answer": "Friction is the force that slows motion when two surfaces rub together."}
]

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
    return generate_with_logging(judge_prompt).text

def run_evaluation():
    print("\n================= Evaluation Pipeline =================\n")
    for i, sample in enumerate(dataset, 1):
        query = sample["query"]
        expected = sample["expected_answer"]
        
        # Model response
        actual_response = generate_with_logging(query).text
        
        # Judge evaluation
        evaluation = judge_response(query, expected, actual_response)
        
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
    
    # Run Evaluation Pipeline
    run_evaluation()
