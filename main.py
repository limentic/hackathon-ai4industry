from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import uuid
import os
from PIL import Image
from transformers import pipeline
import json

app = FastAPI()

# Serve static files at the root URL for index.html
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Global processing queue and WebSocket manager
processing_queue = asyncio.Queue()
clients = {}

# Initialize the VQA pipeline
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-capfilt-large")

# Dummy image processing function
async def process_image(file_path):
    image = Image.open(file_path)

    # Define the questions to ask
    questions = {
        "environnement": "What is the environment in the picture?",
        "subject": "What are the living beings do we see in the image?",
        "action": "What is the subject doing?",
        "number of subject(s)": "How many living beings are in the picture?",
        "people(s) in picture": "Area there any people in the picture?",
    }

    # Store the answers
    answers = {}

    for category, question in questions.items():
        answer = vqa_pipeline(image, question, top_k=1)[0]['answer']
        answers[category] = answer

    os.remove(file_path)

    return json.dumps(answers)

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    clients[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        del clients[client_id]

# Image upload endpoint
@app.post("/upload/")
async def upload_image(file: UploadFile, client_id: str = Form(...)):
    file_id = str(uuid.uuid4())
    file_path = f"uploads/{file_id}_{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    # Add file to processing queue
    await processing_queue.put((file_id, file_path, client_id))
    return JSONResponse({"status": "uploaded", "file_id": file_id})

# Background task to process queue
async def process_queue_task():
    while True:
        file_id, file_path, client_id = await processing_queue.get()
        result = await process_image(file_path)
        
        # Notify the correct WebSocket client
        if client_id in clients:
            await clients[client_id].send_json({"status": "processed", "file_id": file_id})
        
        # Save result to a file
        os.makedirs("results", exist_ok=True)
        with open(f"results/{file_id}.txt", "w") as f:
            f.write(result)

# Run queue processing as a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue_task())

# Result fetch endpoint
@app.get("/result/{file_id}")
async def fetch_result(file_id: str):
    try:
        file_path = f"results/{file_id}.txt"
        with open(file_path, "r") as f:
            file_content = f.read()
            result_json = json.loads(file_content)  # Convert stringified JSON to JSON object
        
        os.remove(file_path)
        
        return {"file_id": file_id, "result": result_json}
    except FileNotFoundError:
        return JSONResponse({"error": "Result not found"}, status_code=404)

# Redirect root URL to index.html
@app.get("/")
async def read_root():
    return JSONResponse({"message": "Redirecting to index.html"}, headers={"Location": "/static/index.html"}, status_code=307)