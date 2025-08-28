# main.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

"""
====================================================
   KalviMentor_AI - Prompt Engineering & Evaluation
====================================================

Includes:
1. Zero-Shot Prompting
2. One-Shot Prompting
3. Multi-Shot Prompting
4. Dynamic Prompting
5. Chain-of-Thought Prompting
6. Evaluation Pipeline with Judge Prompt
7. Token Logging
8. Temperature Control
9. Top-P (Nucleus Sampling) Control
10. Top-K Sampling Control
11. Stop Sequences Control
12. Structured Output Control  ✅

====================================================

🔹 Structured Output:
- Ensures the model replies in a fixed format (e.g., JSON).
- Useful for APIs, chatbots, or any application where the output
  must be machine-readable and not free-form text.

====================================================
"""

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini Model with configs
generation_config = {
    "temperature": 0.7,
    "max_output_tokens": 500,
    "top_p": 0.85,
    "top_k": 40,
    "stop_sequences": ["###"]
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
# Structured Output Example
# ==================================================
def structured_output():
    """
    Ask model to return answer in structured JSON format.
    """
    prompt = """
    You are KalviMentor_AI, a helpful tutor.
    Provide answers ONLY in the following JSON format:
    {
      "question": "<repeat the question>",
      "answer": "<short and clear answer>"
    }

    Question: "What is the capital of France?" ###
    """

    response = generate_with_logging(prompt)
    return response


# ==================================================
# Run All Demonstrations
# ==================================================
if __name__ == "__main__":
    print("\n========== ZERO SHOT ==========")
    print(generate_with_logging("Explain Newton's Second Law in simple terms. ###").text)

    print("\n========== ONE SHOT ==========")
    print(generate_with_logging("Q: What is gravity?\nA: ###").text)

    print("\n========== MULTI SHOT ==========")
    print(generate_with_logging("Q: Who wrote 'Romeo and Juliet'? ###").text)

    print("\n========== DYNAMIC PROMPT ==========")
    print(generate_with_logging("Explain Machine Learning simply. ###").text)

    print("\n========== CHAIN OF THOUGHT ==========")
    print(generate_with_logging("If a car travels at 60 km/h for 2 hours, how far does it go? Let's reason step by step. ###").text)

    print("\n========== STRUCTURED OUTPUT ==========")
    print(structured_output().text)
