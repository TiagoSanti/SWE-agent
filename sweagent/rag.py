import os
import pathspec
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import ollama

extensions = ('.py', '.js', '.java', '.cpp', '.c', '.h', '.md')

# Function to read .gitignore file and return a pathspec matcher
def load_gitignore(clone_dir):
    gitignore_path = os.path.join(clone_dir, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            gitignore_patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
    return pathspec.PathSpec([])

# Step 1: Extract file contents from the cloned repository
def extract_file_contents(clone_dir):
    print("Extracting file contents...")
    file_contents = {}
    gitignore_spec = load_gitignore(clone_dir)
    
    for root, _, files in os.walk(clone_dir):
        # Compute the relative path from the repo root to the current directory
        rel_root = os.path.relpath(root, clone_dir)
        
        # Skip directories listed in .gitignore and common virtual environment directories
        if gitignore_spec.match_file(rel_root) or 'venv' in rel_root or 'env' in rel_root:
            continue
        
        for file in files:
            file_path = os.path.join(root, file)
            # Skip files listed in .gitignore
            if gitignore_spec.match_file(os.path.relpath(file_path, clone_dir)):
                continue
            
            if file.endswith(extensions):
                print(f"Reading file: {file} ...")
                with open(file_path, 'r', errors='ignore') as f:
                    file_contents[file_path] = f.read()
    print(f"Extracted contents of {len(file_contents)} files.")
    return file_contents

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(f"Using device: {device}")

# Initialize the retrieval models
print("Loading DPR models...")
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Function to encode context documents
def encode_contexts(contexts, device):
    print("Encoding context documents...")
    context_embeddings = []
    for i, context in enumerate(contexts):
        if i % 10 == 0:
            print(f"Encoding document {i+1}/{len(contexts)}...")
        inputs = context_tokenizer(context, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        try:
            embeddings = context_encoder(**inputs).pooler_output
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at document {i+1}, switching to CPU for remaining documents.")
            device = 'cpu'
            context_encoder.to(device)
            question_encoder.to(device)
            inputs = inputs.to(device)
            embeddings = context_encoder(**inputs).pooler_output
        context_embeddings.append(embeddings)
        torch.cuda.empty_cache()  # Clear the cache to free up memory
    print("Finished encoding context documents.")
    context_embeddings = [embedding.to(device) for embedding in context_embeddings]
    return torch.cat(context_embeddings), device

# Step 2: Retrieve top N relevant files
def retrieve_contexts(issue_description, file_contents, device, k=5):
    print("Retrieving top relevant files...")
    inputs = question_tokenizer(issue_description, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    question_embedding = question_encoder(**inputs).pooler_output
    context_embeddings, device = encode_contexts(list(file_contents.values()), device)
    question_embedding = question_embedding.to(device)
    scores = torch.matmul(question_embedding, context_embeddings.T)
    top_k_indices = torch.topk(scores, k, dim=1).indices[0]
    
    top_files = [(list(file_contents.keys())[i], scores[0, i].item()) for i in top_k_indices]
    print("Retrieved top relevant files.")
    return top_files

# Function to generate a brief summary of file contents using the Ollama model
def brief_description(file_content):
    prompt = f"Generate a brief summary for the following file content:\n\n{file_content}"
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    summary = response['message']['content']
    return summary

# Main function to perform RAG and get top N files
def rag_top_files(clone_dir, issue_description, top_n=5):
    print("Starting RAG process...")
    file_contents = extract_file_contents(clone_dir)
    top_files = retrieve_contexts(issue_description, file_contents, device, k=top_n)
    
    results = []
    for i, (file, score) in enumerate(top_files):
        print(f"Processing file {i+1}/{len(top_files)}: {file}")
        relative_path = os.path.relpath(file, clone_dir)
        brief_desc = brief_description(file_contents[file])
        results.append((relative_path, brief_desc, score))
    
    print("RAG process completed.")
    return results

clone_dir = os.path.expanduser('~/ais/AIS_mock')
issue_description = """**Summary**
The decision tree image is not being generated when I select the 'decision_tree' model for training using the machine learning endpoint or pipeline endpoint of onca-pintada project. The training completes successfully, and other models produce the expected outputs, but the decision tree visualization is missing.

**Steps to Reproduce**
1. Start the server and navigate to the /machine_learning/{dataset_id}/{file_name}/{classifier} endpoint.
2. Send a GET request with the classifier set to 'decision_tree'.
3. Observe that the model trains successfully, but no image is generated or saved.

**Expected Behavior**
When the decision tree model is selected, the API should generate and save a visualization of the trained decision tree to the specified location.

**Actual Behavior**
The decision tree model trains successfully, and performance metrics are saved, but the visualization image is not created or saved, and there is no indication of an error or failure in this process."""
top_n = 5

top_files_with_descriptions = rag_top_files(clone_dir, issue_description, top_n)
for file, desc, score in top_files_with_descriptions:
    print(f"Score: {score}\nFile: {file}\nDescription: {desc}\n\n")
