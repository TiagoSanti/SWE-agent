import os
import pathspec
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

extensions = ('.py', '.md')

# Function to read .gitignore file and return a pathspec matcher
def load_gitignore(clone_dir):
    gitignore_path = os.path.join(clone_dir, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            gitignore_patterns = f.read().splitlines()
        return pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
    return pathspec.PathSpec([])  # Return an empty PathSpec if no .gitignore is found

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the retrieval models
print("Loading DPR models...")
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Initialize the summarization model
print("Loading summarization model...")
summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
summarizer_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function to encode context documents
def encode_contexts(contexts):
    print("Encoding context documents...")
    context_embeddings = []
    for i, context in enumerate(contexts):
        if i % 10 == 0:
            print(f"Encoding document {i+1}/{len(contexts)}...")
        inputs = context_tokenizer(context, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        embeddings = context_encoder(**inputs).pooler_output
        context_embeddings.append(embeddings)
    print("Finished encoding context documents.")
    return torch.cat(context_embeddings)

# Step 2: Retrieve top N relevant files
def retrieve_contexts(issue_description, file_contents, k=5):
    print("Retrieving top relevant files...")
    inputs = question_tokenizer(issue_description, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    question_embedding = question_encoder(**inputs).pooler_output
    context_embeddings = encode_contexts(list(file_contents.values()))
    
    scores = torch.matmul(question_embedding, context_embeddings.T)
    top_k_indices = torch.topk(scores, k, dim=1).indices[0]
    
    top_files = [(list(file_contents.keys())[i], scores[0, i].item()) for i in top_k_indices]
    print("Retrieved top relevant files.")
    return top_files

# Function to generate a brief summary of file contents
def brief_description(file_content):
    # Use the first few lines of the file as a fallback summary
    first_few_lines = '\n'.join(file_content.split('\n')[:5])
    inputs = summarizer_tokenizer(file_content, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = summarizer.generate(inputs['input_ids'], max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    if not summary.strip():
        return first_few_lines.strip() + '...'
    
    return summary

# Main function to perform RAG and get top N files
def rag_top_files(clone_dir, issue_description, top_n=5):
    print("Starting RAG process...")
    file_contents = extract_file_contents(clone_dir)
    top_files = retrieve_contexts(issue_description, file_contents, k=top_n)
    
    results = []
    for i, (file, score) in enumerate(top_files):
        print(f"Processing file {i+1}/{len(top_files)}: {file}")
        relative_path = os.path.relpath(file, clone_dir)
        brief_desc = brief_description(file_contents[file])
        results.append((relative_path, brief_desc, score))
    
    print("RAG process completed.")
    return results

# Example usage
clone_dir = r'S:\Dev\AIS\AIS_mock'
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
