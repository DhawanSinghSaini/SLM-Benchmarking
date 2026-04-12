import ollama
import json
import csv
import os
import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# Structure of response
RESPONSE_SCHEMA = {
    "answer"    : "string — the actual response to the prompt",
    "confidence": "float between 0.0 and 1.0",
    "category"  : "string — one of: code, translation, writing, qa, documentation",
    "word_count": "integer — approximate word count of answer",
    "summary"   : "string — one sentence summary of the answer"
}

# Example of a valid response
EXAMPLE_RESPONSE = {
    "answer"    : "def factorial(n): return 1 if n==0 else n * factorial(n-1)",
    "confidence": 0.95,
    "category"  : "code",
    "word_count": 12,
    "summary"   : "A recursive Python function that calculates factorial."
}

MODELS = ["phi4-mini", "llama3.2", "gemma2:2b"]

TEST_PROMPTS = [
    "Write a Python function to find the factorial of a number.",
    "Translate 'Nice to meet you' in French.",
    "Draft an email to HR manager for leave approval.",
    "What is the difference between RAM and ROM?",
    "Summarize the movie Oppenheimer in 3 sentences."
]

# System prompt — instructs model to always return JSON
SYSTEM_PROMPT = """
You are a helpful AI assistant.
You must ALWAYS respond in valid JSON format and nothing else.
Do not include any text before or after the JSON.
Do not use markdown code blocks.

Your response must follow this exact structure:
{
    "answer": "<your full answer here>",
    "confidence": <float between 0.0 and 1.0>,
    "category": "<one of: code, translation, writing, qa, documentation>",
    "word_count": <integer>,
    "summary": "<one sentence summary>"
}
"""

TEMPERATURES = [0.0, 0.3, 0.5, 0.7]

TEMP_TEST_PROMPTS = [
    "Write a Python function to find the factorial of a number.",
    "Translate 'Nice to meet you' in French.",
    "Draft an email to HR manager for leave approval.",
    "What is the difference between RAM and ROM?",
    "Summarize the movie Oppenheimer in 3 sentences."
]

# ── CSV fieldnames defined once — used across multiple functions ──
STRUCTURED_FIELDNAMES = [
    "model", "prompt", "category", "confidence",
    "word_count", "summary", "status", "timestamp"
]

TEMPERATURE_FIELDNAMES = [
    "model", "temperature", "prompt", "category",
    "confidence", "word_count", "summary", "status", "timestamp"
]

# Pydantic model — defines what a valid response looks like
class AssistantResponse(BaseModel):
    answer    : str
    confidence: float = Field(ge=0.0, le=1.0)  # must be between 0.0 and 1.0
    category  : Literal["code", "translation", "writing", "qa", "documentation"]
    word_count: int
    summary   : str

def get_structured_response(model_name, user_prompt, temperature=0.7):
    """Get a raw JSON response string from the model"""
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # passing response style
            {"role": "user",   "content": user_prompt}     # passing user prompt
        ],
        format="json",          # forces JSON output mode in Ollama
        options={
            "temperature": temperature
        }
    )
    return response["message"]["content"]

def clean_response(raw):
    """Remove markdown code blocks if model adds them despite instructions"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[-2] if "```" in raw else raw
        raw = raw.removeprefix("json").strip()
    return raw

def parse_and_validate(raw_response):
    """Parse JSON string and validate with Pydantic"""
    try:
        # Step 1 — Clean markdown wrappers if present
        cleaned = clean_response(raw_response)

        # Step 2 — Parse JSON string into dictionary
        data = json.loads(cleaned)

        # Step 3 — Validate with Pydantic
        validated = AssistantResponse(**data)

        print("  Validation : PASSED")
        return validated

    except json.JSONDecodeError as e:
        print(f"  Validation : FAILED — Invalid JSON: {e}")
        return None

    except ValidationError as e:
        print(f"  Validation : FAILED — Schema mismatch: {e}")
        return None

def get_structured_response_with_retry(model_name, user_prompt, temperature=0.7, max_retries=2):
    """Try to get a valid structured response, retry once if it fails"""
    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt} (temp={temperature})...")

        raw    = get_structured_response(model_name, user_prompt, temperature)
        result = parse_and_validate(raw)

        if result:
            return result  # success — return immediately

        if attempt < max_retries:
            print("  Retrying with stronger instruction...")
            user_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your response MUST be valid JSON only. "
                "No extra text. No markdown. Just the JSON object."
            )

    print("  Both attempts failed — skipping this prompt.")
    return None

def save_structured_results(all_results):
    """Save structured output results to CSV"""
    os.makedirs("results", exist_ok=True)
    filename  = "results/structured_results.csv"
    file_exists = os.path.exists(filename)

    file   = open(filename, mode="a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=STRUCTURED_FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    writer.writerows(all_results)
    file.close()
    print(f"Structured results saved to {filename}")

def run_structured_test():
    """Run structured output test on all models and all prompts"""
    all_results = []

    for model in MODELS:
        print(f"\nModel: {model}")
        print("-" * 40)

        for prompt in TEST_PROMPTS:
            print(f"  Prompt: {prompt[:45]}...")

            result = get_structured_response_with_retry(model, prompt)

            if result:
                print(f"  Category  : {result.category}")
                print(f"  Confidence: {result.confidence}")
                print(f"  Words     : {result.word_count}")
                print(f"  Summary   : {result.summary[:60]}...")

                all_results.append({
                    "model"     : model,
                    "prompt"    : prompt[:50] + "...",
                    "category"  : result.category,
                    "confidence": result.confidence,
                    "word_count": result.word_count,
                    "summary"   : result.summary,
                    "status"    : "PASSED",
                    "timestamp" : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                })
            else:
                all_results.append({
                    "model"     : model,
                    "prompt"    : prompt[:50] + "...",
                    "category"  : "N/A",
                    "confidence": 0,
                    "word_count": 0,
                    "summary"   : "FAILED",
                    "status"    : "FAILED",
                    "timestamp" : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                })
            print()

    save_structured_results(all_results)
    print("\nStructured output test complete!")

def analyse_results():
    """Read structured_results.csv and print pass/fail rates per model"""

    # Fix 2 — check file exists before opening
    if not os.path.exists("results/structured_results.csv"):
        print("  No results file found. Run structured test first.")
        return

    results = {}

    file   = open("results/structured_results.csv", "r")
    reader = csv.DictReader(file)

    for row in reader:
        model  = row["model"]
        status = row["status"]
        if model not in results:
            results[model] = {"PASSED": 0, "FAILED": 0}
        results[model][status] += 1

    file.close()    # Fix 1 — close the file

    print("\n" + "=" * 50)
    print("VALIDATION PASS/FAIL RATES")
    print("=" * 50)

    for model, counts in results.items():
        total = counts["PASSED"] + counts["FAILED"]
        rate  = round((counts["PASSED"] / total) * 100, 1)
        print(f"\nModel  : {model}")
        print(f"Passed : {counts['PASSED']}/{total}")
        print(f"Failed : {counts['FAILED']}/{total}")
        print(f"Rate   : {rate}%")
        print("-" * 30)

def run_temperature_experiment():
    """Run same prompts at different temperatures and record results"""
    all_Results = []
    os.makedirs("results", exist_ok=True)

    for model in MODELS:
        print(f"\nModel: {model}")
        print("=" * 50)

        for temp in TEMPERATURES:
            print(f"\n  Temperature: {temp}")
            print("  " + "-" * 40)

            for prompt in TEMP_TEST_PROMPTS:
                print(f"    Prompt: {prompt[:40]}...")

                result = get_structured_response_with_retry(
                    model, prompt, temperature=temp
                )

                if result:
                    print(f"    Confidence : {result.confidence}")   # Fix 3 — typo fixed
                    print(f"    Word Count : {result.word_count}")
                    print(f"    Summary    : {result.summary[:50]}...")

                    all_Results.append({
                        "model"      : model,
                        "temperature": temp,
                        "prompt"     : prompt[:50] + "...",
                        "category"   : result.category,
                        "confidence" : result.confidence,
                        "word_count" : result.word_count,
                        "summary"    : result.summary,
                        "status"     : "PASSED",
                        "timestamp"  : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    })
                else:
                    all_Results.append({
                        "model"      : model,
                        "temperature": temp,
                        "prompt"     : prompt[:50] + "...",
                        "category"   : "N/A",
                        "confidence" : 0,
                        "word_count" : 0,
                        "summary"    : "FAILED",
                        "status"     : "FAILED",
                        "timestamp"  : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    })
                print()

    # Save temperature experiment results
    filename = "results/temperature_results.csv"
    file     = open(filename, "w", newline="", encoding="utf-8")

    # Fix 4 — use explicit fieldnames instead of all_Results[0].keys()
    writer = csv.DictWriter(file, fieldnames=TEMPERATURE_FIELDNAMES)
    writer.writeheader()
    if all_Results:                 # only write rows if there is data
        writer.writerows(all_Results)
    file.close()

    print(f"\nTemperature results saved to {filename}")
    return all_Results

def summarise_temperature_variance(all_results):
    """Print how word count and confidence vary per model per temperature"""
    print("\n" + "=" * 60)
    print("TEMPERATURE VARIANCE SUMMARY")
    print("=" * 60)

    for model in MODELS:
        print(f"\nModel: {model}")
        print("-" * 40)

        for temp in TEMPERATURES:
            runs = [
                r for r in all_results
                if r["model"]       == model
                and r["temperature"] == temp
                and r["status"]      == "PASSED"
            ]

            if runs:
                avg_conf  = round(sum(r["confidence"]  for r in runs) / len(runs), 3)
                avg_words = round(sum(r["word_count"]   for r in runs) / len(runs), 1)
                pass_rate = f"{len(runs)}/{len(TEMP_TEST_PROMPTS)}"

                print(f"  Temp {temp} → "
                      f"Avg Confidence: {avg_conf} | "
                      f"Avg Words: {avg_words} | "
                      f"Pass: {pass_rate}")
            else:
                print(f"  Temp {temp} → All failed")

if __name__ == "__main__":
    run_structured_test()
    analyse_results()
    results = run_temperature_experiment()
    summarise_temperature_variance(results)