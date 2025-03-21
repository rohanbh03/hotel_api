from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time

# --- 1. Initialize the FastAPI app ---
app = FastAPI(
    title="Hotel Booking Analytics & Q&A API with Real-time Updates",
    description="API to serve hotel booking analytics and RAG-based question answering with SQLite integration.",
    version="2.0"
)

# --- 2. Initialize SQLite Database ---
DB_PATH = "hotel_bookings.db"

# Connect to SQLite and create tables
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table for hotel bookings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT,
            reservation_status_date TEXT,
            stays_in_weekend_nights INTEGER,
            stays_in_week_nights INTEGER,
            lead_time INTEGER,
            market_segment TEXT,
            adr REAL,
            is_canceled INTEGER
        )
    ''')

    # Create table for query history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            response TEXT,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# --- 3. Load Data into SQLite ---
def load_data_into_db():
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_csv("C:/Users/rohan/hotel_api/cleaned_hotel_bookings.csv")

    # Clear existing data
    conn.execute("DELETE FROM bookings")

    # Insert new data
    data.to_sql("bookings", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

# --- 4. Retrieve Data from SQLite ---
def get_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM bookings", conn)
    conn.close()
    return df

# --- 5. RAG Setup with FAISS ---
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data = get_data_from_db()

# Convert rows to natural language text
def convert_to_text(row):
    return (
        f"Booking from {row['country']} on {row['reservation_status_date']}. "
        f"Stayed {row['stays_in_weekend_nights']} weekend nights and {row['stays_in_week_nights']} weekday nights. "
        f"Lead time: {row['lead_time']} days. "
        f"Market Segment: {row['market_segment']}. "
        f"ADR: {row['adr']}. "
        f"Status: {'Cancelled' if row['is_canceled'] == 1 else 'Confirmed'}."
    )

data['text'] = data.apply(convert_to_text, axis=1)

# Create FAISS index
embeddings = model.encode(data['text'].tolist(), convert_to_tensor=False)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# LLM Pipeline
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# --- 6. Models for API ---
class Query(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    report_type: str

# --- 7. RAG Retrieval Function ---
def retrieve_answer(question, top_k=5):
    question_embedding = model.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)

    relevant_texts = [data['text'][i] for i in indices[0]]
    context = "\n".join(relevant_texts)

    # Generate answer
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm(
        prompt,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        truncation=True,
        pad_token_id=50256
    )[0]['generated_text']

    # Store query and response in query history
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO query_history (question, response, timestamp) VALUES (?, ?, ?)",
                   (question, result, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return result

# --- 8. Analytics Endpoint ---
@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    report_type = request.report_type.lower()
    df = get_data_from_db()

    if report_type == "revenue_trend":
        trend = df.groupby('reservation_status_date')['adr'].sum().to_dict()
        return {"Revenue Trend": trend}

    elif report_type == "cancellation_rate":
        total = len(df)
        cancelled = len(df[df['is_canceled'] == 1])
        rate = (cancelled / total) * 100
        return {"Cancellation Rate (%)": round(rate, 2)}

    elif report_type == "geographical_distribution":
        distribution = df['country'].value_counts().to_dict()
        return {"Geographical Distribution": distribution}

    elif report_type == "lead_time_distribution":
        lead_time_dist = df['lead_time'].describe().to_dict()
        return {"Lead Time Distribution": lead_time_dist}

    else:
        raise HTTPException(status_code=400, detail="Invalid report type")

# --- 9. Question Answering Endpoint ---
@app.post("/ask")
async def ask_question(query: Query):
    answer = retrieve_answer(query.question)
    return {"answer": answer}

# --- 10. Query History Endpoint ---
@app.get("/history")
async def get_query_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_history")
    rows = cursor.fetchall()
    conn.close()

    history = [{"id": row[0], "question": row[1], "response": row[2], "timestamp": row[3]} for row in rows]
    return {"query_history": history}

# --- 11. Health Check Endpoint ---
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "faiss": "ok",
        "database": "connected"
    }

# --- 12. Initialize Database on Startup ---
init_db()
load_data_into_db()
