import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Define the device for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data and model only once
@st.cache_resource
def load_resources():
    # Load the CSV file
    data = pd.read_csv('python_code.csv')
    questions = data['question'].tolist()
    solutions = data['solution'].tolist()
    
    # Load the embedder model
    embedder_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Encode the questions into embeddings
    question_embeddings = embedder_model.encode(questions, convert_to_tensor=True)
    
    # Create FAISS index for retrieval
    d = question_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(question_embeddings.cpu()))
    
    # Load the CodeT5 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
    codet5_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-large-ntp-py").to(device)
    
    return embedder_model, question_embeddings, index, tokenizer, codet5_model, solutions, questions

# Load resources
embedder_model, question_embeddings, index, tokenizer, codet5_model, solutions, questions = load_resources()

# Function to retrieve the nearest question based on input
def retrieve_answer(input_question, questions):
    input_embedding = embedder_model.encode([input_question], convert_to_tensor=True)
    input_embedding = np.array(input_embedding.cpu())
    D, I = index.search(input_embedding, k=1)
    distance = D[0][0]
    
    if distance < 0.5:  # Adjust threshold as needed
        return solutions[I[0][0]], questions[I[0][0]]
    else:
        return None, None

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.cuda.amp.autocast():
        generated_tokens = codet5_model.generate(**inputs, max_length=200)
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def rag_generate(input_question):
    retrieved_solution, matched_question = retrieve_answer(input_question, questions)

    if retrieved_solution is not None:
        prompt = f"# Input Question: {input_question}\n# Context: {retrieved_solution}\n"
        generated_code = generate_code(prompt)
        return {
            'input_question': input_question,
            'matched_question': matched_question,
            'retrieved_solution': retrieved_solution,
            'generated_code': generated_code
        }
    else:
        prompt = f"# Input Question: {input_question}\n"
        generated_code = generate_code(prompt)
        return {
            'input_question': input_question,
            'error': "No relevant questions found in our     dataset.",
            'generated_code': generated_code
        }

# Streamlit UI
st.title("Code Generation with RAG")
st.write("Enter your programming question below:")

user_input = st.text_input("Your Question:")

if st.button("Generate Code"):
    if user_input:
        result = rag_generate(user_input)
        if 'error' not in result:
            st.subheader("Results")
            st.write(f"**Input Question:** {result['input_question']}")
            st.write(f"**Matched Question:** {result['matched_question']}")
            st.write(f"**Retrieved Solution:** {result['retrieved_solution']}")
            st.subheader("Generated Code:")
            st.code(result['generated_code'])
        else:
            st.error(result['error'])
            st.write(f"**Input Question:** {result['input_question']}")
            st.subheader("Generated Code:")
            st.code(result['generated_code'])
    else:
        st.warning("Please enter a question.")
