#📘 NTSE_QA Benchmark Evaluation – Gemma 3 4B 📘

---

## ASSIGNMENT OVERVIEW

The task was to evaluate the `NTSE_QA.csv` benchmark using the **Gemma3 4B** model (Ollama).
The primary objectives included:

- Establishing a baseline accuracy using a fixed prompt.

- Applying different improvement techniques such as **Prompt Engineering, Retrieval-Augmented Generation (RAG) and Self-Consistency Decoding**.

- Running all experiments offline in Ollama.

- Comparing results across techniques and analyzing why each method worked (or didn’t).

---

## APPROACH

### Step 1 – Baseline Evaluation

Loaded the NTSE_QA.csv dataset.

Used Gemma 3 4B with a fixed system prompt.

Recorded initial accuracy (48%).

### Step 2 – Improvement Techniques

1. Prompt Engineering → Refined the prompt, with step-by-step logic, added few-shot examples.

2. Retrieval-Augmented Generation (RAG) → Retrieved similar solved questions before answering, better context.

3. Self-Consistency Decoding → Generated multiple responses (5) and used majority voting.

4. Better Error Handling & Logging to avoid script failures.

### Step 3 – Comparison

Accuracy of each technique was measured and compared against the baseline.

Observed which strategies contributed most to performance gains.

---

## PERFORMANCE RESULTS
### Technique   -   Accuracy (%)
- **Baseline (Fixed Prompt)** -   48.00
- **Prompt Engineering**  -   	51.06
- **Retrieval-Augmented Generation (RAG)**    -   	54.43
- **Self-Consistency Decoding**   -   	50.43

---

## KEY INSIGHTS

- **Context-Aware Responses**: RAG provided the **highest gain (+6.43%)**, as injecting relevant examples improved context understanding.

- Prompt Engineering improved clarity and boosted performance slightly.

- **More Robust Predictions**: Self-Consistency Decoding added robustness as Majority voting reduces randomness but showed limited improvement compared to RAG.

- This assignment demonstrates how structured techniques—prompting, retrieval, and reasoning agents—can significantly **improve the accuracy of medium-scale LLMs in multiple-choice QA tasks** making model more reliable and optimized.

## FUTURE WORK
Testing with larger models like GPT-4 or fine-tuning Gemma.
 



