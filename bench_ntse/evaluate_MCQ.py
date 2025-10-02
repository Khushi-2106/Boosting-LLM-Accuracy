import os 
from langchain_ollama import ChatOllama
import pandas as pd
import json
from tqdm import tqdm


syst_prompt = """
Please provide the correct answer to the following multiple-choice question. The answers must be in a JSON format as follows:

Outpur format Json : 


{
"correct_option": "<answer id>"

}



Make sure to use the format above for the answer the given question and include the correct option as 'correct_option'  .
"""

#load the questions
questions = pd.read_csv("NTSE_QA.csv")

def evaluate(model):
    print(f"Evaluating model: {model}")

    save_path_csv = os.path.join(model + "_NTSE_QA_answers_.csv")  

    # for open a models replace with ChatOpenAI


    llm =  ChatOllama(model=model, temperature=0,format="json")
    
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses)
        
    else:
        start = 1
    print(f"Starting from {start} ....")

    for idx, row in tqdm(questions.iloc[start-1:].iterrows()):
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                user_prompt = f"""
                Question id {idx}\n
                Question: {row["Question"]}\n
                Option 1: {row["Option A"]}\n
                Option 2: {row["Option B"]}\n
                Option 3: {row["Option C"]}\n
                Option 4: {row["Option D"]}\n\n
                """

                # Get the correct answer and the correct option
                correct_option = row["Correct Answer"]

                # Create the message list to pass into the LLM
                messages = [
                    ("system", syst_prompt),
                    ("human", user_prompt),
                ]

                # Invoke the LLM to get the answer in JSON format
                predicted_answer = llm.invoke(messages).content

                # Convert the predicted answer string into JSON
                predicted_json = json.loads(predicted_answer)

                # Extract the predicted answer and option from the JSON
                llm_answer_option = predicted_json["correct_option"]

                # Check if the LLM's predicted answer matches the correct answer
   
                if llm_answer_option == correct_option:
                    answered_status = "Right"

                else:
                    answered_status = "Wrong"

                # Prepare the output to be saved
                output = {
                    "Question id": idx,
                    "Question": row["Question"],
                    "Option 1": row["Option A"],
                    "Option 2": row["Option B"],
                    "Option 3": row["Option C"],
                    "Option 4": row["Option D"],
                    "Correct Option": correct_option,
                    "Generated Option":  llm_answer_option ,
                    "Truth": answered_status,

                }

                # Convert to DataFrame and append to CSV
                output_df = pd.DataFrame([output])
                output_df.to_csv(save_path_csv, mode='a', header=not pd.io.common.file_exists(save_path_csv), index=False)

                # If successful, exit the retry loop
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error type: {type(e).__name__} on attempt {attempt}")
                print(f"Error message: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for Question id {idx}. Skipping to next.")



if __name__ == "__main__":
    models = ["gemma3:4b"]
    for model in models:
        evaluate(model)


       
