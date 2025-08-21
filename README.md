# KalviMentor_AI

KalviMentor_AI is an AI-powered **Educational Mentor Agent** designed to guide students in their academic journey.  
It provides **personalized learning support**, practice questions, assessment evaluation, and performance tracking using advanced prompting techniques and LLM features.

---

## 🚀 Project Description
KalviMentor_AI acts as a **virtual mentor** for students.  
It leverages **Zero-shot, One-shot, Multi-shot, Dynamic, and Chain-of-thought prompting techniques** to deliver accurate, adaptive, and structured responses.  
The project integrates **evaluation pipelines, embeddings, vector databases, and similarity functions** to improve personalized learning experiences.

---

## ✨ Features
- Personalized subject guidance  
- Practice & quiz generation  
- Evaluation framework with test datasets  
- Semantic search with embeddings + vector database  
- Function calling for enhanced interactivity  
- Token usage logging for cost monitoring  
- Configurable Temperature, Top-P, Top-K & Stop Sequences  
- Structured outputs for consistent answers  

---

## 📂 Project Structure
```
KalviMentor_AI/
│── README.md
│── prompts/
│   ├── system_prompt.txt
│   ├── user_prompts/
│   │   ├── zero_shot.txt
│   │   ├── one_shot.txt
│   │   ├── multi_shot.txt
│   │   ├── dynamic_prompt.txt
│   │   └── chain_of_thought.txt
│── evaluation/
│   ├── dataset.json
│   ├── judge_prompt.txt
│   ├── test_framework.py
│── ai_config/
│   ├── temperature_config.json
│   ├── top_p_config.json
│   ├── top_k_config.json
│   ├── stop_sequence.json
│── embeddings/
│   ├── vector_store.db
│   ├── similarity.py
│── src/
│   ├── mentor_agent.py
│   ├── function_calls.py
│   └── server.py
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/KalviMentor_AI.git
cd KalviMentor_AI
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API Key
Add your **Gemini API key** in `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

### Run AI Server
```bash
python src/server.py
```

### Access AI Mentor
Open browser and visit:
```
http://localhost:8000
```

---

## 🧠 Prompt Engineering

- **System Prompt**: Defines KalviMentor_AI’s role as a friendly educational mentor.  
- **Zero Shot Prompting**: Directly asks questions without examples.  
- **One Shot Prompting**: Provides a single example before user query.  
- **Multi Shot Prompting**: Uses multiple examples to improve accuracy.  
- **Dynamic Prompting**: Adjusts based on student’s level and progress.  
- **Chain of Thought Prompting**: Encourages step-by-step reasoning for problem-solving.  

---

## 📊 Evaluation Framework
- **Dataset**: Includes at least 5 queries with expected answers.  
- **Judge Prompt**: Evaluates correctness, clarity, and relevance.  
- **Testing Framework**: Runs all test cases and logs results.  

---

## 🔬 LLM Fine Controls
- **Tokens**: Logs token usage after each call.  
- **Temperature**: Adjusts creativity of responses.  
- **Top-P & Top-K**: Controls randomness and diversity of outputs.  
- **Stop Sequence**: Ends response at defined marker.  
- **Structured Output**: Ensures consistent JSON/Markdown formatting.  

---

## 📚 Embeddings & Vector Search
- Embedding generation for student queries.  
- Vector Database for semantic search.  
- **Cosine Similarity, Euclidean Distance, and Dot Product** used for comparison.  

---

## 🎯 Future Enhancements
- Mobile App Integration  
- Voice-based mentor interaction  
- Multi-language student support  
- Real-time notifications & reminders  

---

## 🏆 Project Goal
To build an **AI Mentor Agent** that **empowers students** with personalized learning, structured practice, and continuous feedback.

---

## 📜 License
MIT License © 2025 KalviMentor_AI
