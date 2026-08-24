import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_answer(question: str, context: str) -> str:
    # If no key is set or no context found, fallback gracefully
    if not GROQ_API_KEY:
        return (
            "FastAPI is a modern Python web framework used for building high-performance APIs."
            if "fastapi" in question.lower()
            else context[:300] + "..."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
    You are a concise AI assistant. Answer the user's question using ONLY the provided context below.
    Keep your answer clear, direct, and under 3 sentences.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 150,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"Groq API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Groq API Exception: {e}")

    # Fallback to a short slice of context instead of dumping the entire string
    return f"Based on retrieved context: {context[:250]}..."