import os
import json
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
from rag_helper import retrieve_similar_questions  # Import RAG function

# Load model
model_name = "gemma3:4b"
llm = ChatOllama(model=model_name, temperature=0, format="json")

# Load dataset
questions = pd.read_csv("NTSE_QA.csv")

# 🔹 Improved System Prompt
syst_prompt = """
You are an AI assistant trained to solve multiple-choice questions (MCQs) with logical reasoning.  
Always follow these structured steps before answering:  

1.**Understand the Question**: Identify the key concept and required knowledge.  
2.**Analyze Answer Choices**: Consider each option carefully.  
3.**Eliminate Incorrect Answers**: Use elimination techniques.  
4.**Select the Best Option**: Choose the most logical answer.  
5.*Strict JSON Output**: Do NOT explain, only return the answer in JSON format.  

### Example:
```json
{
  "correct_option": "C"
}
Failure to return a valid JSON format will result in rejection. """

def evaluate(): 
    print(f"Total questions in dataset: {len(questions)}")
    save_path_csv = os.path.join(model_name + "_NTSE_QA_answers_.csv")

if os.path.exists(save_path_csv):
    responses = pd.read_csv(save_path_csv)
    start = len(responses)
else:
    start = 1

print(f"Starting from {start} ...")
correct_count = 0
total_count = 0

for idx, row in tqdm(questions.iloc[start-1:].iterrows()):
    attempt = 0
    success = False

    while attempt < 3 and not success:
        try:
            # 🔹 Retrieve similar questions using RAG
            retrieved_info = retrieve_similar_questions(row["Question"]).strip()

            # 🔹 Few-Shot Learning Examples for Better Accuracy
            few_shot_examples = """
                Example 1:
                Question: What is the capital of France? A) Berlin
                B) Madrid
                C) Paris
                D) Rome
                Correct Answer: {"correct_option": "C"}

                Example 2:
                Question: What is 5 + 7? A) 10
                B) 11
                C) 12
                D) 13
                Correct Answer: {"correct_option": "C"}

                Now, analyze the following question: 
                """
            # 🔹 Enhanced User Prompt
            user_prompt = f"""{retrieved_info} # Inject similar questions (RAG) {few_shot_examples}

                Question ID {idx}:
                Question: {row["Question"]}
                A) {row["Option A"]}
                B) {row["Option B"]}
                C) {row["Option C"]}
                D) {row["Option D"]}

                Return the answer in strict JSON format. """
            correct_option = row["Correct Answer"].strip()

            # 🔹 Improved Self-Consistency Decoding: Majority Voting
            answer_options = []
            for _ in range(5):  # Generate 5 responses
                try:
                    predicted_answer = llm.invoke([
                        ("system", syst_prompt),
                        ("human", user_prompt)
                    ]).content

                    predicted_json = json.loads(predicted_answer)

                    if "correct_option" in predicted_json:
                        answer_options.append(predicted_json["correct_option"])
                    else:
                        print(f"Invalid JSON format for Question ID {idx}. Skipping response.")

                except json.JSONDecodeError:
                    print(f"Decoding error for Question ID {idx}. Skipping response.")

            # 🔹 Majority Voting with Fallback
            if answer_options:
                final_answer = max(set(answer_options), key=answer_options.count)
            else:
                final_answer = "Unknown"  # Fallback if no valid answer is found

            answered_status = "Right" if final_answer == correct_option else "Wrong"

            # 🔹 Save results
            output = {
                "Question ID": idx,
                "Question": row["Question"],
                "Correct Option": correct_option,
                "Generated Option": final_answer,
                "Truth": answered_status,
            }

            output_df = pd.DataFrame([output])
            output_df.to_csv(save_path_csv, mode='a', header=not os.path.exists(save_path_csv), index=False)

            if answered_status == "Right":
                correct_count += 1
            total_count += 1

            success = True  # Exit loop on success

        except Exception as e:
            attempt += 1
            print(f"Error on attempt {attempt}: {e}")
            if attempt == 3:
                print(f"Failed after 3 attempts for Question ID {idx}. Skipping.")

if total_count > 0:
    print(f"Accuracy = {correct_count / total_count:.2%}")
else:
    print("No valid questions were processed.")

if __name__ == "main": 
    evaluate()
