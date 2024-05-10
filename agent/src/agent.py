from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import numpy as np
import faiss
import litellm
from litellm import completion
from dotenv import load_dotenv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
main_dir = os.path.abspath(__file__)
for i in range(0, 3):
    main_dir = os.path.dirname(main_dir)

src_dir = os.path.join(main_dir, "agent", "src")
data_dir = os.path.join(main_dir, "agent", "data")

# Load API keys
load_dotenv(os.path.join(src_dir, ".env"))
apikey = os.getenv('OPENAI_API_KEY')

# Initialize DPR encoders and tokenizers
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

# List of context sentences
contexts = ["Hello world", "Machine learning is fun", "I like to eat pizza", "The weather is sunny today", "BERT is a transformer model that I like more than pizza."]

# Function to encode contexts
def encode_contexts(contexts, model, tokenizer):
    model.eval()
    with torch.no_grad():
        return torch.stack([model(**tokenizer(context, return_tensors="pt", padding=True, truncation=True)).pooler_output for context in contexts])

# Encode contexts
context_embeddings = encode_contexts(contexts, context_encoder, context_tokenizer).numpy().squeeze()

# Create a FAISS index (Flat index for simplicity)
d = context_embeddings.shape[1]  # Dimension of embeddings
print(f"Dimensions: {d}")
index = faiss.IndexFlatL2(d)
print("Shape of context_embeddings:", context_embeddings.shape)
index.add(context_embeddings)  # Add context embeddings to the index

# Function to encode a query and retrieve the most relevant contexts
def retrieve_context(query, q_model, q_tokenizer, index, contexts):
    q_model.eval()
    with torch.no_grad():
        query_emb = q_model(**q_tokenizer(query, return_tensors="pt", padding=True, truncation=True)).pooler_output.numpy()
    distances, indices = index.search(query_emb, 2)  # Retrieve the top 1 closest context
    return contexts[indices[0][0]], contexts[indices[0][1]]

# Example query
query = input("Ask Question: \n")

# Retrieve the most relevant context
relevant_context = retrieve_context(query, question_encoder, question_tokenizer, index, contexts)
print("Most relevant context:", relevant_context)

response = completion(
    api_key=apikey,
    base_url="https://drchat.xyz",
    model = "gpt-3.5-turbo-16k",
    custom_llm_provider="openai",
    messages = [{ "content": f"""
                Given the context: {relevant_context}
                Answer the question: {query}
                ""","role": "user"}],
    temperature=0.5
)

print(response.choices[0].message.content)