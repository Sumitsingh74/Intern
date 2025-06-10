# from dotenv import load_dotenv
# load_dotenv(dotenv_path=r"C:\Users\asus\OneDrive\Desktop\online_interview_system\.env")
# load_dotenv()

import os
import re
import json
import time
import asyncio
from turtle import st
from typing import List, Dict
from retry import retry

from openai import OpenAI  # Used for OpenRouter integration

# ------------------- API Key Configuration ------------------- #

OPENROUTER_API_KEY = "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"

if not OPENROUTER_API_KEY:
    print("[ERROR] OPENROUTER_API_KEY not found")
else:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


# ------------------- Utility ------------------- #

def extract_numeric_score(text: str) -> float:
    match = re.search(r"\b(10(?:\.0)?|[0-9](?:\.[0-9])?)\b", text)
    return float(match.group()) if match else -1.0


# ------------------- Scoring Engines ------------------- #

MODELS = {
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gemini-pro": "google/gemini-pro-1.5",
    "llama-3.3": "meta-llama/llama-3.3-8b-instruct:free",
    "claude-opus": "anthropic/claude-opus-4"
}


@retry(tries=3, delay=1)
def score_response(question: str, answer: str, model_name: str) -> float:
    try:
        prompt = f"""
You are an expert technical interviewer. Evaluate the candidate response below.

Question: {question}
Answer: {answer}

Provide a numeric score between 0-10. Return only the number."""

        response = client.chat.completions.create(
            model=MODELS[model_name],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        return extract_numeric_score(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[{model_name} Error] {e}")
        return -1.0


# ------------------- Orchestrator ------------------- #

def score_all_responses(candidate_id: str) -> List[Dict]:
    candidate_dir = os.path.join("interviews", candidate_id)
    filepath = os.path.join(candidate_dir, "responses.json")

    if not os.path.exists(filepath):
        print(f"[ERROR] responses.json not found for candidate: {candidate_id}")
        return []

    with open(filepath, "r") as f:
        responses = json.load(f)

    results = []
    model_mapping = {
        "gpt-4-turbo": "gpt_4_score",
        "gemini-pro": "gemini_1.5_score",
        "claude-opus": "opus_4_score",
        "llama-3.3": "llama_3.3_score"
    }

    for res in responses:
        question = res.get("question", "").strip()
        answer = res.get("transcript", "").strip()

        if not answer:
            print(f"[Skipped] Empty answer for question: {question}")
            continue

        scores = {}
        valid_scores = []

        # Score with all models (without st.spinner)
        for model_name in model_mapping.keys():
            print(f"Scoring with {model_name}...")
            score = score_response(question, answer, model_name)
            display_key = model_mapping[model_name]
            scores[display_key] = score
            if score >= 0:
                valid_scores.append(score)

        # Calculate average
        average = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else -1.0

        results.append({
            "question": question,
            "answer": answer,
            **scores,
            "average_score": average
        })

    # Save results
    scored_filepath = os.path.join(candidate_dir, "scored_responses.json")
    with open(scored_filepath, "w") as f:
        json.dump(results, f, indent=4)

    return results