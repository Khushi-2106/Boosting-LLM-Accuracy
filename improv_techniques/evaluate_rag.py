import os
import json
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model for answer generation
model_name = "gemma3:4b"
llm = ChatOllama(model=model_name, temperature=0, format="json")

# Load your MCQ dataset
questions = pd.read_csv("NTSE_QA.csv")

# Define system prompt for reasoning
syst_prompt = """
You are an expert AI designed to answer multiple-choice questions with logical reasoning.
Follow this step-by-step process:

1. Analyze the question and each option.
2. Eliminate incorrect choices.
3. Select the most logical answer.
4. Return the answer in JSON format.

Example:
{
  "correct_option": "B"
}

Only return a valid JSON response. No explanations.
"""

# Retrieve similar questions using Cosine Similarity
def retrieve_similar_questions(current_question, questions_df, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions_df['Question'])  # Vectorize all questions
    current_tfidf = vectorizer.transform([current_question])  # Vectorize the current question
    cosine_sim = cosine_similarity(current_tfidf, tfidf_matrix)  # Compute cosine similarity
    similar_indices = cosine_sim.argsort()[0][-top_n:]  # Get indices of the most similar questions

    # Fetch similar questions
    similar_questions = []
    for idx in similar_indices:
        similar_questions.append(f"Question: {questions_df.iloc[idx]['Question']} \nOptions: {questions_df.iloc[idx]['Option A']}, {questions_df.iloc[idx]['Option B']}, {questions_df.iloc[idx]['Option C']}, {questions_df.iloc[idx]['Option D']}")
    
    return "\n\n".join(similar_questions)

def evaluate():
    # Check if the responses CSV exists to determine the starting point
    save_path_csv = os.path.join(model_name + "_NTSE_QA_answers_.csv")

    # Initialize start index
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses) + 1  # Start from the next unprocessed question
    else:
        start = 1  # Start from the first question if no previous responses exist

    print(f"Starting from question index {start} ...")

    correct_count = 0
    total_count = 0

    # Iterate through the questions and apply RAG
    for idx, row in tqdm(questions.iloc[start-1:].iterrows(), total=len(questions)-start+1):
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                # Retrieve similar questions from the MCQ dataset using cosine similarity
                retrieved_info = retrieve_similar_questions(row["Question"], questions, top_n=3)

                # Construct the prompt for the generative model
                user_prompt = f"""
                {retrieved_info}  # Inject similar questions from the MCQ dataset
                Question ID {idx}:
                Question: {row["Question"]}
                A) {row["Option A"]}
                B) {row["Option B"]}
                C) {row["Option C"]}
                D) {row["Option D"]}
                """

                correct_option = row["Correct Answer"].strip()

                # Generate answers using the LLM (RAG)
                predicted_answer = llm.invoke([
                    ("system", syst_prompt),
                    ("human", user_prompt)
                ]).content

                # print(f"Predicted Answer (raw): {predicted_answer}")

                try:
                    predicted_json = json.loads(predicted_answer)
                    llm_answer_option = predicted_json["correct_option"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error decoding JSON or missing 'correct_option' for Question ID {idx}: {e}")
                    llm_answer_option = "Unknown"

                # Normalize the answers (remove spaces, parentheses, and make everything lowercase)
                def normalize_answer(answer):
                    return answer.strip().replace("(", "").replace(")", "").lower()

                # Normalize both the generated answer and the correct option
                llm_answer_option_normalized = normalize_answer(llm_answer_option)
                correct_option_normalized = normalize_answer(correct_option)

                # Compare the normalized answers
                answered_status = "Right" if llm_answer_option_normalized == correct_option_normalized else "Wrong"


                # Save results
                output = {
                    "Question ID": idx,
                    "Question": row["Question"],
                    "Correct Option": correct_option,
                    "Generated Option": llm_answer_option,
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
                print(f"Error on attempt {attempt} for Question ID {idx}: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for Question ID {idx}. Skipping.")

    # Calculate and print accuracy
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"Accuracy = {accuracy:.2%}")
    else:
        print("No questions were processed. Accuracy could not be calculated.")

if __name__ == "__main__":
    evaluate()
