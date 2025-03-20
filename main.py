from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics import accuracy_score
from difflib import SequenceMatcher
import time

# --- 1. Initialize the FastAPI app ---
app = FastAPI(
    title="Hotel Booking Analytics & Q&A API",
    description="API to serve hotel booking analytics and RAG-based question answering.",
    version="1.0"
)

# --- 2. Load and Prepare Data ---
# Load cleaned dataset
file_path = "C:\\Users\\rohan\\hotel_api\\cleaned_hotel_bookings.csv"
data = pd.read_csv(file_path)

# Convert rows to natural language text
def convert_to_text(row):
    return (
        f"Booking made from {row['country']} on {row['reservation_status_date']}. "
        f"Stayed for {row['stays_in_weekend_nights']} weekend nights and {row['stays_in_week_nights']} weekday nights. "
        f"Lead time: {row['lead_time']} days. "
        f"Market Segment: {row['market_segment']}. "
        f"ADR: {row['adr']}. "
        f"Booking status: {'Cancelled' if row['is_canceled'] == 1 else 'Confirmed'}."
    )

data['text'] = data.apply(convert_to_text, axis=1)

# --- 3. Create FAISS Index with Clustering ---
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(data['text'].tolist(), convert_to_tensor=False)

dimension = embeddings.shape[1]
nlist = 100  # Number of clusters

# Create the FAISS index with clustering
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# Train the index
index.train(np.array(embeddings))
index.add(np.array(embeddings))

# --- 4. LLM Pipeline for Q&A ---
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# --- 5. Define Models for API Request ---
class Query(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    report_type: str

# --- 6. RAG Retrieval Function ---
def retrieve_answer(question, top_k=5):
    """Retrieve relevant booking data using FAISS and generate an answer with LLM"""
    question_embedding = model.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_texts = [data['text'][i] for i in indices[0]]
    context = "\n".join(relevant_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm(
        prompt,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        truncation=True,
        pad_token_id=50256
    )[0]['generated_text']
    return result

# --- 7. Accuracy Evaluation ---
def evaluate_accuracy(question, ground_truth, top_k=5):
    """Evaluate accuracy of RAG system by comparing generated answers with ground truth."""
    generated_answer = retrieve_answer(question, top_k)
    similarity = SequenceMatcher(None, generated_answer, ground_truth).ratio()
    print(f"Question: {question}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Similarity Score: {round(similarity * 100, 2)}%")
    return similarity

# --- 8. Analytics Endpoint ---
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    report_type = request.report_type.lower()
    if report_type == "revenue_trend":
        trend = data.groupby('reservation_status_date')['adr'].sum().to_dict()
        return {"Revenue Trend": trend}
    elif report_type == "cancellation_rate":
        total = len(data)
        cancelled = len(data[data['is_canceled'] == 1])
        rate = (cancelled / total) * 100
        return {"Cancellation Rate (%)": round(rate, 2)}
    elif report_type == "geographical_distribution":
        distribution = data['country'].value_counts().to_dict()
        return {"Geographical Distribution": distribution}
    elif report_type == "lead_time_distribution":
        lead_time_dist = data['lead_time'].describe().to_dict()
        return {"Lead Time Distribution": lead_time_dist}
    else:
        raise HTTPException(status_code=400, detail="Invalid report type")

# --- 9. Question Answering Endpoint with Response Time ---
@app.post("/ask")
async def ask_question(query: Query):
    start_time = time.time()
    answer = retrieve_answer(query.question)
    end_time = time.time()
    response_time = round(end_time - start_time, 4)
    return {
        "answer": answer,
        "response_time_seconds": response_time
    }

# --- 10. Root Endpoint ---
@app.get("/")
async def home():
    return {"message": "Welcome to the Hotel Booking Analytics API! Use /docs to access Swagger UI."}

if __name__ == "__main__":
    print("\n--- Accuracy Test ---\n")

    # Example test cases
    questions = [
        "What is the average price of a hotel booking?",
        "Which locations had the highest booking cancellations?"
    ]
    
    ground_truths = [
        "The average price of a hotel booking is approximately $120.",
        "The locations with the highest booking cancellations are Spain and Portugal."
    ]

    for q, gt in zip(questions, ground_truths):
        score = evaluate_accuracy(q, gt)
        print(f"Accuracy for '{q}': {round(score * 100, 2)}%")


