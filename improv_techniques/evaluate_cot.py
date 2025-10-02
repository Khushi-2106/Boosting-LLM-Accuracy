import os 
from langchain_ollama import ChatOllama
import pandas as pd
import json
from tqdm import tqdm

def evaluate(model):
    print(f"Evaluating model: {model}")
    save_path_csv = os.path.join(model + "_NTSE_QA_CoT_answers_.csv")  
    
    llm = ChatOllama(model=model, temperature=0, format="json")
    
    # Load questions
    questions = pd.read_csv("NTSE_QA.csv")
    
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses)
    else:
        start = 1
    
    print(f"Starting from {start} ...")
    count = 0
    t_count = 0

    for idx, row in tqdm(questions.iloc[start-1:].iterrows()):
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                # Chain-of-Thought Prompting
                syst_prompt = """
                Please provide the correct answer to the following multiple-choice question.
                Think step by step and reason through the answer before selecting the correct option.
                The output must be in JSON format:

                {
                    "reasoning": "<step-by-step explanation>",
                    "correct_option": "<answer id>"
                }
                """
                
                user_prompt = f"""
                Question ID: {idx}
                Question: {row["Question"]}
                Option A: {row["Option A"]}
                Option B: {row["Option B"]}
                Option C: {row["Option C"]}
                Option D: {row["Option D"]}
                
                Think step by step before choosing the correct answer.
                """
                
                correct_option = row["Correct Answer"].strip()

                messages = [("system", syst_prompt), ("human", user_prompt)]
                
                # Invoke the LLM with CoT prompting
                predicted_answer = llm.invoke(messages).content
                predicted_json = json.loads(predicted_answer)
                llm_answer_option = predicted_json.get("correct_option", "None")
                reasoning = predicted_json.get("reasoning", "No explanation provided.")

                #Normalize the answers (remove spaces, parentheses, and make everything lowercase)
                def normalize_answer(answer):
                    return answer.strip().replace("(", "").replace(")", "").lower()

                # Normalize both the generated answer and the correct option
                llm_answer_option_normalized = normalize_answer(llm_answer_option)
                correct_option_normalized = normalize_answer(correct_option)

                # Compare the normalized answers
                answered_status = "Right" if llm_answer_option_normalized == correct_option_normalized else "Wrong"
                
                if answered_status == "Right":
                    count += 1
                t_count += 1
                
                # Prepare output
                output = {
                    "Question ID": idx,
                    "Question": row["Question"],
                    "Option A": row["Option A"],
                    "Option B": row["Option B"],
                    "Option C": row["Option C"],
                    "Option D": row["Option D"],
                    "Correct Option": correct_option,
                    "Generated Option": llm_answer_option,
                    "Truth": answered_status,
                    "Reasoning": reasoning
                }
                
                # Save results
                output_df = pd.DataFrame([output])
                output_df.to_csv(save_path_csv, mode='a', header=not os.path.exists(save_path_csv), index=False)
                
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error type: {type(e).__name__} on attempt {attempt}")
                print(f"Error message: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for Question ID {idx}. Skipping to next.")
    
    print(f"Final Accuracy = {count/t_count:.2%}")

if __name__ == "__main__":
    models = ["gemma3:4b"]
    for model in models:
        evaluate(model)
