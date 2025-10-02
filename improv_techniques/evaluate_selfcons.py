import os
from collections import Counter
from langchain_ollama import ChatOllama
import pandas as pd
import json
from tqdm import tqdm

def evaluate(model, num_samples=5):
    print(f"Evaluating model: {model}")

    save_path_csv = os.path.join(model + "_NTSE_QA_answers_.csv")  
    llm = ChatOllama(model=model, temperature=0.7, format="json")
    
    questions = pd.read_csv("NTSE_QA.csv")
    
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses)
    else:
        start = 1
    
    print(f"Starting from {start} ....")
    correct_count = 0
    total_count = 0
    
    for idx, row in tqdm(questions.iloc[start-1:].iterrows()):
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                user_prompt = f"""
                Question ID: {idx}\n
                Question: {row['Question']}\n
                Option A: {row['Option A']}\n
                Option B: {row['Option B']}\n
                Option C: {row['Option C']}\n
                Option D: {row['Option D']}\n\n
                Please provide the correct answer in JSON format:\n
                {{"correct_option": "<answer id>"}}
                """
                
                correct_option = row["Correct Answer"].strip()
                
                predictions = []
                
                for _ in range(num_samples):
                    messages = [("system", "Answer the question correctly in JSON format."),
                                ("human", user_prompt)]
                    
                    predicted_answer = llm.invoke(messages).content
                    predicted_json = json.loads(predicted_answer)
                    predictions.append(predicted_json.get("correct_option", ""))
                
                # Self-Consistency Decoding (Majority Vote)
                most_common_answer, _ = Counter(predictions).most_common(1)[0]
                
                # Evaluate correctness
                if most_common_answer == correct_option:
                    answered_status = "Right"
                    correct_count += 1
                else:
                    answered_status = "Wrong"
                
                total_count += 1
                
                # Prepare output to be saved
                output = {
                    "Question ID": idx,
                    "Question": row["Question"],
                    "Option A": row["Option A"],
                    "Option B": row["Option B"],
                    "Option C": row["Option C"],
                    "Option D": row["Option D"],
                    "Correct Option": correct_option,
                    "Generated Option": most_common_answer,
                    "Truth": answered_status,
                    "All Predictions": ", ".join(predictions)
                }
                
                output_df = pd.DataFrame([output])
                output_df.to_csv(save_path_csv, mode='a', header=not os.path.exists(save_path_csv), index=False)
                
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error type: {type(e).__name__} on attempt {attempt}")
                print(f"Error message: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for Question ID {idx}. Skipping to next.")
    
    print(f"Accuracy = {correct_count / total_count:.2%}")

if __name__ == "__main__":
    models = ["gemma3:4b"]
    for model in models:
        evaluate(model, num_samples=5)
