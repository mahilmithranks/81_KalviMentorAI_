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
8. Temperature Control
9. Top-P (Nucleus Sampling) Control

====================================================

🔹 Tokens:
- Tokens are chunks of text (words/subwords).
- Models process text in tokens. More tokens = more cost & time.

🔹 Temperature:
- Controls creativity/randomness.
- Low (0.2) → focused, deterministic.
- High (0.8) → creative, diverse.

🔹 Top-P (Nucleus Sampling):
- Chooses from top tokens whose cumulative probability ≥ P.
- Low (0.5) → safe, predictable.
- High (0.9) → diverse, creative.

====================================================
"""

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini Model with Temperature + Top P
generation_config = {
    "temperature": 0.7,   # creativity control
    "max_output_tokens": 500,
    "top_p": 0.85,        # nucleus sampling
    "top_k": 40
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)


# ==================================================
# Utility: Generate with token logging
# ==================================================
def generate_with_logging(prompt):
    """Generate model output and log token usage."""
    response = model.generate_content(prompt)

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print(f"🔹 Tokens Used - Prompt: {usage.prompt_token_count}, "
              f"Candidates: {usage.candidates_token_count}, "
              f"Total: {usage.total_token_count}\n")
    else:
        print("⚠️ Token usage metadata not available.\n")

    return response


# ==================================================
# Prompting Examples
# ==================================================

def zero_shot():
    prompt = """
    You are KalviMentor_AI, a friendly mentor.
    Explain Newton’s Second Law of Motion in simple terms.
    """
    return generate_with_logging(prompt)


def one_shot():
    prompt = """
    You are KalviMentor_AI, a friendly mentor.

    Example Q&A:
    Q: What is photosynthesis?
    A: Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.

    Now answer this:
    Q: What is gravity?
    """
    return generate_with_logging(prompt)


def multi_shot():
    prompt = """
    You are KalviMentor_AI, a friendly mentor.

    Example 1:
    Q: What is the capital of France?
    A: Paris.

    Example 2:
    Q: What is the square root of 16?
    A: 4.

    Now answer this:
    Q: Who wrote 'Romeo and Juliet'?
    """
    return generate_with_logging(prompt)


def dynamic_prompt(student_name="Mahil", subject="Python"):
    prompt = f"""
    You are KalviMentor_AI, helping a student named {student_name}.
    The student asks: "Can you explain {subject} in simple terms?"
    Provide a friendly, structured answer.
    """
    return generate_with_logging(prompt)


def chain_of_thought():
    prompt = """
    You are KalviMentor_AI, a reasoning assistant.
    A student asks: "If a car travels at 60 km/h for 2 hours, how far does it go?"

    Let's reason step by step (do not show reasoning, only final answer):
    """
    return generate_with_logging(prompt)


# ==================================================
# Evaluation Pipeline
# ==================================================

dataset = [
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of India?", "expected": "New Delhi"},
    {"input": "Who wrote Hamlet?", "expected": "William Shakespeare"},
    {"input": "Square root of 81?", "expected": "9"},
    {"input": "Chemical symbol of water?", "expected": "H2O"},
]

def evaluate_model():
    print("\n🔹 Running Evaluation Pipeline...\n")
    results = []
    for sample in dataset:
        user_prompt = f"You are KalviMentor_AI. Answer clearly: {sample['input']}"
        response = generate_with_logging(user_prompt)

        judge_prompt = f"""
        You are a strict evaluator.
        Compare the model's answer with the expected answer.
        Question: {sample['input']}
        Model Answer: {response.text}
        Expected Answer: {sample['expected']}
        Does the model's answer match the expected one? Reply with Yes or No.
        """
        judge_response = model.generate_content(judge_prompt)
        verdict = judge_response.text.strip()
        results.append({"input": sample["input"], "model_answer": response.text, "expected": sample["expected"], "verdict": verdict})

    print("✅ Evaluation Results:")
    for r in results:
        print(r)


# ==================================================
# Run All Demonstrations
# ==================================================

if __name__ == "__main__":
    print("\n========== ZERO SHOT ==========")
    print(zero_shot().text)

    print("\n========== ONE SHOT ==========")
    print(one_shot().text)

    print("\n========== MULTI SHOT ==========")
    print(multi_shot().text)

    print("\n========== DYNAMIC PROMPT ==========")
    print(dynamic_prompt(student_name="Ananya", subject="Machine Learning").text)

    print("\n========== CHAIN OF THOUGHT ==========")
    print(chain_of_thought().text)

    print("\n========== EVALUATION ==========")
    evaluate_model()
