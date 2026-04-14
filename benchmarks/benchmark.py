import ollama   #library for chatting with model
import time     #library to handle time
import csv      #library to handle data read and write in csv
import os       #library for file and directory operations
import datetime
import psutil


MODELS = ["phi4-mini", "llama3.2", "gemma2:2b"] #available models

# Sample Prompts
TEST_PROMPTS = [
    # SET 1
    "Write Java code to reverse a singly linked list.",     # Code Assistance
    "Translate 'Hi, I am an AI Chatbot' to Hindi.",         # Translation
    "Draft an email to HR manager for leave approval.",     # Writing Help
    "Explain types of ML learning techniques.",             # QnA
    "Document pros and cons of EVs.",                       # Documentation

    # SET 2
    "Write a Python function to find the factorial of a number.",   # Code Assistance
    "Translate 'Nice to meet you' in French.",                      # Translation
    "Write a cover letter for a software engineering internship.",  # Writing Help
    "What is the difference between RAM and ROM?",                  # QnA
    "Explain the movie 'Oppenheimer' in 500 words.",                # Documentation
]

# Function to return total system RAM currently in use (always positive)
def get_memory_usage_mb():
    ram = psutil.virtual_memory()
    return round(ram.used / 1024 / 1024, 2)

# Function used for benchmarking.
def benchmark_model(model_name, prompt):

    # --- Peak memory tracker ---
    peak_memory = 0

    # --- Total latency start ---
    total_start = time.time()                 # Time when function starts

    # --- Stream response to capture TTFT ---
    first_token_time = None
    full_response = ""
    token_count = 0

    stream = ollama.chat(
        model = model_name,
        messages = [{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        # Capture time to first token
        if first_token_time is None:
            first_token_time = time.time() - total_start

        # Track peak memory on every chunk
        current_mem = get_memory_usage_mb()
        if current_mem > peak_memory:
            peak_memory = current_mem

        token = chunk["message"]["content"]
        full_response += token
        token_count += len(token.split())

    # --- Total latency end ---
    total_latency = time.time() - total_start

    # --- Tokens per second ---
    token_per_second = token_count / total_latency if total_latency > 0 else 0
    # if latency is 0: response generated in no time — practically impossible, may happen on error
    # if latency is not 0: tps = tokens / latency

    # Function returns all benchmark stats for given model and prompt
    return {
        "model"                 : model_name,                               # model name
        "prompt"                : prompt[:50] + "...",                      # input prompt (truncated)
        "ttft_seconds"          : round(first_token_time, 3),               # indicates responsiveness
        "total_latency_seconds" : round(total_latency, 3),                  # indicates overall performance
        "tokens_per_second"     : round(token_per_second, 2),               # indicates speed of generation
        "response_length"       : len(full_response),                       # indicates length of response
        "peak_memory_mb"        : peak_memory,                              # peak system RAM during inference
        "timestamp"             : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    }

# Function used for writing to csv
def save_result(results, filename="results/benchmark_results.csv"):
    os.makedirs("results", exist_ok=True)                   # creates folder in case missing
    fieldnames = [
        "model", "prompt", "ttft_seconds", "total_latency_seconds",
        "tokens_per_second", "response_length", "peak_memory_mb", "timestamp"
    ]
    file_exists = os.path.exists(filename)

    file = open(filename, mode="a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not file_exists:                                     # if file missing, write header first
        writer.writeheader()
    writer.writerows(results)
    print(f"Results saved to {filename}")
    file.close()

# Function used to run all models and all prompts
def run_all_benchmarks():
    all_Results = []                                        # list to store all outputs

    for model in MODELS:                                    # iterate over models
        print(f"\nBenchmarking: {model}")
        print("-" * 40)

        for prompt in TEST_PROMPTS:                         # iterate over prompts
            print(f"Running : {prompt[:50]}...")
            try:
                result = benchmark_model(model, prompt)
                all_Results.append(result)
                print(f"  TTFT: {result['ttft_seconds']}s | "
                      f"Latency: {result['total_latency_seconds']}s | "
                      f"Speed: {result['tokens_per_second']} tok/s | "
                      f"Peak Mem: {result['peak_memory_mb']} MB")
            except Exception as e:
                print(f"  ERROR on {model}: {e}")
                continue

    save_result(all_Results)
    save_summary(all_Results)
    print("\nAll benchmarks complete!")
    print_summary(all_Results)
    return all_Results

# Function to print average metrics per model in terminal
def print_summary(all_Result):
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY - AVERAGE PER MODEL")
    print("=" * 60)

    for model in MODELS:
        model_result = [r for r in all_Result if r["model"] == model]

        avg_ttft    = sum(r["ttft_seconds"]          for r in model_result) / len(model_result)
        avg_latency = sum(r["total_latency_seconds"]  for r in model_result) / len(model_result)
        avg_speed   = sum(r["tokens_per_second"]      for r in model_result) / len(model_result)
        avg_mem     = sum(r["peak_memory_mb"]         for r in model_result) / len(model_result)

        print(f"\nModel          : {model}")
        print(f"Avg TTFT       : {round(avg_ttft, 3)}s")
        print(f"Avg Latency    : {round(avg_latency, 3)}s")
        print(f"Avg Speed      : {round(avg_speed, 2)} tok/s")
        print(f"Avg Peak Memory: {round(avg_mem, 2)} MB")
        print("-" * 40)

# Function to print a single result in terminal
def print_result(result):
    print(f"  TTFT        : {result['ttft_seconds']}s")
    print(f"  Latency     : {result['total_latency_seconds']}s")
    print(f"  Speed       : {result['tokens_per_second']} tok/s")
    print(f"  Response    : {result['response_length']} chars")
    print(f"  Peak Memory : {result['peak_memory_mb']} MB")
    print()

# Function to save averages per model to summary CSV
def save_summary(all_Result):
    summary = []
    for model in MODELS:
        model_results = [r for r in all_Result if r["model"] == model]

        summary.append({
            "model"                  : model,
            "avg_ttft_seconds"       : round(sum(r["ttft_seconds"]          for r in model_results) / len(model_results), 3),
            "avg_latency_seconds"    : round(sum(r["total_latency_seconds"]  for r in model_results) / len(model_results), 3),
            "avg_tokens_per_second"  : round(sum(r["tokens_per_second"]      for r in model_results) / len(model_results), 2),
            "avg_peak_memory_mb"     : round(sum(r["peak_memory_mb"]         for r in model_results) / len(model_results), 2),
        })

    file = open("results/summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
    file.close()
    print("Summary saved to results/summary.csv")

if __name__ == "__main__":
    run_all_benchmarks()