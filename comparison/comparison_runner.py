import ollama
import csv
import os
import time
import datetime
import psutil

MODELS = ["phi4-mini", "llama3.2", "gemma2:2b"]

CATEGORIES = {
    "code"         : list(range(0,  6)),
    "translation"  : list(range(6,  12)),
    "writing"      : list(range(12, 18)),
    "qa"           : list(range(18, 24)),
    "documentation": list(range(24, 30)),
}

TEST_PROMPTS = [
    # ── CATEGORY 1: CODE ASSISTANCE (6 prompts) ──
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a string.",
    "Write a Java program to find the largest element in an array.",
    "Write a SQL query to find the second highest salary from an employee table.",
    "Explain what a binary search algorithm is and write it in Python.",
    "Write a Python function to count the frequency of each word in a string.",

    # ── CATEGORY 2: TRANSLATION (6 prompts) ──
    "Translate 'Good morning, how are you?' to Hindi.",
    "Translate 'I love programming' to French.",
    "Translate 'Artificial intelligence is changing the world' to Spanish.",
    "Translate 'Thank you for your help' to Japanese.",
    "Translate 'What time is it?' to German.",
    "Translate 'I am learning machine learning' to Tamil.",

    # ── CATEGORY 3: WRITING HELP (6 prompts) ──
    "Write a professional email to a client apologising for a project delay.",
    "Write a cover letter for a software engineer applying to a product startup.",
    "Write a LinkedIn post announcing a new job at a tech company.",
    "Write a resignation letter to a manager with a 2-week notice.",
    "Write a thank you email after a job interview.",
    "Write a short bio for a data science student's portfolio website.",

    # ── CATEGORY 4: Q&A (6 prompts) ──
    "What is the difference between machine learning and deep learning?",
    "Explain how a transformer model works in simple terms.",
    "What is the CAP theorem in distributed systems?",
    "What is the difference between supervised and unsupervised learning?",
    "What is overfitting in machine learning and how do you prevent it?",
    "What is the difference between a process and a thread?",

    # ── CATEGORY 5: DOCUMENTATION (6 prompts) ──
    "Summarize the pros and cons of using microservices architecture.",
    "Document the key differences between REST and GraphQL APIs.",
    "Summarize what Docker does and why developers use it.",
    "Document the SOLID principles of software design in brief.",
    "Explain the differences between SQL and NoSQL databases.",
    "Summarize what Agile methodology is and its core principles.",
]

# Scoring rubric reference
SCORING_RUBRIC = {
    5: "Excellent — correct, complete, well structured",
    4: "Good — correct and reasonably complete",
    3: "OK — correct but incomplete or poorly structured",
    2: "Partial — missing key information",
    1: "Wrong — incorrect or completely off topic"
}

# CSV fieldnames defined once — used across functions
COMPARISON_FIELDNAMES = [
    "model", "category", "prompt",
    "ttft_seconds", "total_latency_seconds",
    "tokens_per_second", "response_length",
    "peak_memory_mb", "timestamp"
]

# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def get_memory_usage_mb():
    """Returns total system RAM currently in use — always positive"""
    return round(psutil.virtual_memory().used / 1024 / 1024, 2)

def get_category_for_index(i):
    """Return category name for a given prompt index"""
    for cat, indices in CATEGORIES.items():
        if i in indices:
            return cat
    return "unknown"

# ─────────────────────────────────────────────
# PHASE 3: RUN BENCHMARKS
# ─────────────────────────────────────────────

def run_single(model_name, prompt, category):
    """Run a single model on a single prompt and collect all metrics"""
    peak_memory      = 0
    total_start      = time.time()
    first_token_time = None
    full_response    = ""
    token_count      = 0

    stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time() - total_start

        current_mem = get_memory_usage_mb()
        if current_mem > peak_memory:
            peak_memory = current_mem

        token = chunk["message"]["content"]
        full_response += token
        token_count   += len(token.split())

    total_latency     = time.time() - total_start
    tokens_per_second = round(token_count / total_latency, 2) if total_latency > 0 else 0

    return {
        "model"                 : model_name,
        "category"              : category,
        "prompt"                : prompt[:60] + "...",
        "ttft_seconds"          : round(first_token_time, 3),
        "total_latency_seconds" : round(total_latency, 3),
        "tokens_per_second"     : tokens_per_second,
        "response_length"       : len(full_response),
        "peak_memory_mb"        : peak_memory,
        "response"              : full_response,       # stored in memory only, not CSV
        "timestamp"             : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    }

def run_comparison():
    """Run all 3 models on all 30 prompts — 90 total runs"""
    all_results = []

    for model in MODELS:
        print(f"\nModel: {model}")
        print("=" * 50)

        for i, prompt in enumerate(TEST_PROMPTS):
            category = get_category_for_index(i)
            print(f"  [{i+1:02d}/30] [{category}] {prompt[:45]}...")

            try:
                result = run_single(model, prompt, category)
                all_results.append(result)
                print(f"  TTFT: {result['ttft_seconds']}s | "
                      f"Latency: {result['total_latency_seconds']}s | "
                      f"Speed: {result['tokens_per_second']} tok/s | "
                      f"Mem: {result['peak_memory_mb']} MB")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    save_comparison_results(all_results)
    print_comparison_summary(all_results)
    return all_results

# ─────────────────────────────────────────────
# SAVE & LOAD RESULTS
# ─────────────────────────────────────────────

def save_comparison_results(all_results, filename="results/comparison_results.csv"):
    """Save benchmark results to CSV — excludes full response to keep file clean"""
    os.makedirs("results", exist_ok=True)

    file   = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDNAMES)
    writer.writeheader()

    for r in all_results:
        row = {k: r[k] for k in COMPARISON_FIELDNAMES}
        writer.writerow(row)

    file.close()
    print(f"\nComparison results saved to {filename}")

def load_results(filename="results/comparison_results.csv"):
    """Load existing benchmark results from CSV — avoids re-running 90 benchmarks"""
    if not os.path.exists(filename):
        print(f"No results file found at {filename}. Run run_comparison() first.")
        return []

    rows = []
    file   = open(filename, "r")
    reader = csv.DictReader(file)

    for row in reader:
        row["ttft_seconds"]          = float(row["ttft_seconds"])
        row["total_latency_seconds"] = float(row["total_latency_seconds"])
        row["tokens_per_second"]     = float(row["tokens_per_second"])
        row["response_length"]       = int(row["response_length"])
        row["peak_memory_mb"]        = float(row["peak_memory_mb"])
        row["response"]              = ""    # not stored in CSV
        rows.append(row)

    file.close()
    print(f"Loaded {len(rows)} results from {filename}")
    return rows

# ─────────────────────────────────────────────
# PRINT SUMMARIES
# ─────────────────────────────────────────────

def print_comparison_summary(all_results):
    """Print average metrics per model per category"""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY — AVERAGES PER MODEL PER CATEGORY")
    print("=" * 70)

    for model in MODELS:
        print(f"\nModel: {model}")
        print("-" * 50)

        for category in CATEGORIES.keys():
            cat_results = [
                r for r in all_results
                if r["model"] == model and r["category"] == category
            ]
            if cat_results:
                avg_ttft    = round(sum(r["ttft_seconds"]          for r in cat_results) / len(cat_results), 3)
                avg_latency = round(sum(r["total_latency_seconds"]  for r in cat_results) / len(cat_results), 2)
                avg_speed   = round(sum(r["tokens_per_second"]      for r in cat_results) / len(cat_results), 2)

                print(f"  {category:<15} → "
                      f"TTFT: {avg_ttft}s | "
                      f"Latency: {avg_latency}s | "
                      f"Speed: {avg_speed} tok/s")

# ─────────────────────────────────────────────
# MEMORY ANALYSIS
# ─────────────────────────────────────────────

def analyse_memory(all_results):
    """Print average peak memory per model per category"""
    print("\n" + "=" * 60)
    print("PEAK MEMORY USAGE — AVERAGES PER MODEL PER CATEGORY")
    print("=" * 60)

    for model in MODELS:
        print(f"\nModel: {model}")
        print("-" * 40)

        model_mems = []
        for category in CATEGORIES.keys():
            cat_results = [
                r for r in all_results
                if r["model"] == model and r["category"] == category
            ]
            if cat_results:
                avg_mem = round(
                    sum(r["peak_memory_mb"] for r in cat_results)
                    / len(cat_results), 2
                )
                model_mems.append(avg_mem)
                print(f"  {category:<15} → Avg Peak Memory: {avg_mem} MB")

        if model_mems:
            overall = round(sum(model_mems) / len(model_mems), 2)
            print(f"  {'OVERALL':<15} → Avg Peak Memory: {overall} MB")

# ─────────────────────────────────────────────
# QUALITY SCORING
# ─────────────────────────────────────────────

def score_by_category(all_results):
    """Manually score average quality per model per category — rate 1 to 5"""
    print("\n" + "=" * 60)
    print("QUALITY SCORING — Rate each category 1-5")
    print("=" * 60)
    print("Rubric:")
    for score, desc in SCORING_RUBRIC.items():
        print(f"  {score} = {desc}")

    scores = {}

    for model in MODELS:
        scores[model] = {}
        for category in CATEGORIES.keys():
            cat_results = [
                r for r in all_results
                if r["model"] == model and r["category"] == category
            ]

            if not cat_results:
                continue

            print(f"\n{'='*50}")
            print(f"Model    : {model}")
            print(f"Category : {category}")
            print(f"Sample response preview:")
            print(f"  {cat_results[0]['response'][:300]}...")
            print()

            while True:
                try:
                    score = int(input(f"Quality score for {model} — {category} (1-5): "))
                    if 1 <= score <= 5:
                        scores[model][category] = score
                        break
                    print("  Please enter a number between 1 and 5.")
                except ValueError:
                    print("  Please enter a valid number.")

    return scores

def save_quality_scores(scores):
    """Save quality scores to CSV"""
    rows = []
    for model, cats in scores.items():
        for category, score in cats.items():
            rows.append({
                "model"         : model,
                "category"      : category,
                "quality_score" : score,
                "timestamp"     : datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            })

    os.makedirs("results", exist_ok=True)
    file   = open("results/quality_scores.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=["model","category","quality_score","timestamp"])
    writer.writeheader()
    writer.writerows(rows)
    file.close()
    print("\nQuality scores saved to results/quality_scores.csv")

def load_quality_scores(filename="results/quality_scores.csv"):
    """Load existing quality scores from CSV"""
    if not os.path.exists(filename):
        return {}

    scores = {}
    file   = open(filename, "r")
    reader = csv.DictReader(file)

    for row in reader:
        model    = row["model"]
        category = row["category"]
        score    = int(row["quality_score"])
        if model not in scores:
            scores[model] = {}
        scores[model][category] = score

    file.close()
    print(f"Loaded quality scores from {filename}")
    return scores

# ─────────────────────────────────────────────
# MASTER TABLE & WINNERS
# ─────────────────────────────────────────────

def print_master_table(all_results, scores):
    """Print the definitive master comparison table combining all metrics"""

    schema = {
        "phi4-mini" : 73.3,
        "llama3.2"  : 86.7,
        "gemma2:2b" : 100.0
    }

    print("\n" + "=" * 85)
    print("MASTER COMPARISON TABLE — ALL METRICS")
    print("=" * 85)
    print(f"{'Model':<16} {'TTFT':<10} {'Latency':<12} {'Speed':<14} {'Memory(MB)':<14} {'Quality':<10} {'Schema'}")
    print("-" * 85)

    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model]

        avg_ttft    = round(sum(r["ttft_seconds"]          for r in model_results) / len(model_results), 3)
        avg_latency = round(sum(r["total_latency_seconds"]  for r in model_results) / len(model_results), 2)
        avg_speed   = round(sum(r["tokens_per_second"]      for r in model_results) / len(model_results), 2)
        avg_memory  = round(sum(r["peak_memory_mb"]         for r in model_results) / len(model_results), 2)

        model_scores = list(scores.get(model, {}).values())
        avg_quality  = round(sum(model_scores) / len(model_scores), 2) if model_scores else 0

        print(f"{model:<16} {str(avg_ttft)+'s':<10} {str(avg_latency)+'s':<12} "
              f"{str(avg_speed)+' t/s':<14} {str(avg_memory):<14} "
              f"{str(avg_quality)+'/5':<10} {schema[model]}%")

    print("=" * 85)

def save_master_summary(all_results, scores):
    """Save master summary combining all metrics to CSV"""
    schema = {
        "phi4-mini" : 73.3,
        "llama3.2"  : 86.7,
        "gemma2:2b" : 100.0
    }
    rows = []

    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model]

        avg_ttft    = round(sum(r["ttft_seconds"]          for r in model_results) / len(model_results), 3)
        avg_latency = round(sum(r["total_latency_seconds"]  for r in model_results) / len(model_results), 2)
        avg_speed   = round(sum(r["tokens_per_second"]      for r in model_results) / len(model_results), 2)
        avg_memory  = round(sum(r["peak_memory_mb"]         for r in model_results) / len(model_results), 2)

        model_scores = list(scores.get(model, {}).values())
        avg_quality  = round(sum(model_scores) / len(model_scores), 2) if model_scores else 0

        rows.append({
            "model"                  : model,
            "avg_ttft_seconds"       : avg_ttft,
            "avg_latency_seconds"    : avg_latency,
            "avg_tokens_per_second"  : avg_speed,
            "avg_peak_memory_mb"     : avg_memory,
            "avg_quality_score"      : avg_quality,
            "schema_compliance_pct"  : schema[model]
        })

    os.makedirs("results", exist_ok=True)
    file   = open("results/master_summary.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    file.close()
    print("Master summary saved to results/master_summary.csv")

def print_category_winners(all_results, scores):
    """Print the winning model per metric per category"""
    print("\n" + "=" * 60)
    print("CATEGORY WINNERS")
    print("=" * 60)

    for category in CATEGORIES.keys():
        print(f"\nCategory: {category.upper()}")
        print("-" * 40)

        cat_data = {}
        for model in MODELS:
            cat_results = [
                r for r in all_results
                if r["model"] == model and r["category"] == category
            ]
            if cat_results:
                cat_data[model] = {
                    "ttft"   : round(sum(r["ttft_seconds"]         for r in cat_results) / len(cat_results), 3),
                    "latency": round(sum(r["total_latency_seconds"] for r in cat_results) / len(cat_results), 2),
                    "speed"  : round(sum(r["tokens_per_second"]     for r in cat_results) / len(cat_results), 2),
                    "quality": scores.get(model, {}).get(category, 0)
                }

        if cat_data:
            best_ttft    = min(cat_data, key=lambda m: cat_data[m]["ttft"])
            best_latency = min(cat_data, key=lambda m: cat_data[m]["latency"])
            best_speed   = max(cat_data, key=lambda m: cat_data[m]["speed"])
            best_quality = max(cat_data, key=lambda m: cat_data[m]["quality"])

            print(f"  Fastest TTFT    : {best_ttft}  ({cat_data[best_ttft]['ttft']}s)")
            print(f"  Lowest Latency  : {best_latency}  ({cat_data[best_latency]['latency']}s)")
            print(f"  Highest Speed   : {best_speed}  ({cat_data[best_speed]['speed']} tok/s)")
            print(f"  Best Quality    : {best_quality}  ({cat_data[best_quality]['quality']}/5)")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Load or run benchmarks
    results = load_results()
    if not results:
        print("No existing results found. Running full comparison...")
        results = run_comparison()
    else:
        print_comparison_summary(results)

    # Step 2 — Memory analysis
    analyse_memory(results)

    # Step 3 — Load or run quality scoring
    scores = load_quality_scores()
    if not scores:
        print("\nNo quality scores found. Starting scoring session...")
        scores = score_by_category(results)
        save_quality_scores(scores)
    else:
        print("Quality scores loaded from file.")

    # Step 4 — Master table and winners
    print_master_table(results, scores)
    print_category_winners(results, scores)
    save_master_summary(results, scores)

    print("\nDay 7 complete!")