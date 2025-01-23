from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import uuid
import os
from PIL import Image
from transformers import pipeline
import json

# --- New Imports for CNN ---
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

app = FastAPI()

# Serve static files at the root URL for index.html
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Global processing queue and WebSocket manager
processing_queue = asyncio.Queue()
clients = {}

# ---------------------------
# 1) Load the VQA pipeline
# ---------------------------
vqa_pipeline = pipeline(
    "visual-question-answering",
    model="Salesforce/blip-vqa-capfilt-large",
    use_fast=False
)

# ---------------------------
# 2) Load our trained CNN
# ---------------------------
# Adjust num_classes to match what you had during training
num_classes = 6  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the same architecture used during training (ResNet-18 here)
cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)

# Load the saved weights
cnn_model_path = "cnn_model.pth"
if os.path.exists(cnn_model_path):
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    print("CNN model weights loaded from", cnn_model_path)
else:
    print("[WARNING] cnn_model.pth not found. Make sure you have it in the same folder.")

cnn_model.eval()
cnn_model.to(device)

# Load class names
class_names_file = "class_names.txt"
if os.path.exists(class_names_file):
    with open(class_names_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    print("[WARNING] class_names.txt not found.")
    class_names = []

# Define any required transforms for the CNN
cnn_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------------
# Dummy (or new) image processing function with both VQA and CNN classification
# ----------------------------------------------------------------------------
async def process_image(file_path):
    image = Image.open(file_path)

    # --------------------------------------------------------------------
    # A) CNN Inference to classify your custom categories
    # --------------------------------------------------------------------
    # Convert the image for the model
    image_tensor = cnn_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    cnn_predicted_class = class_names[predicted.item()] if class_names else "Unknown"

    # --------------------------------------------------------------------
    # B) BLIP VQA Inference
    # --------------------------------------------------------------------
    questions = {
        "environnement": "What is the environment in the picture?",
        "subject": "What are the living beings do we see in the image?",
        "action": "What is the subject doing?",
        "number of subject(s)": "How many living beings are in the picture?",
        "people(s) in picture": "Are there any people in the picture?",
        "jellyfish": "Are there any jellyfish in the picture?"
    }

    answers = {}
    for category, question in questions.items():
        answer = vqa_pipeline(image, question, top_k=1)[0]['answer']
        answers[category] = answer

    if (answers["jellyfish"] == "yes"):
        species = ""
        for i in range(len(cnn_predicted_class)):
            if cnn_predicted_class[i] == "_":
                species += " "
            else:
                species += cnn_predicted_class[i]

        answers["species"] = species
        answers['subject'] = "jellyfish"
    
    answers.pop("jellyfish")

    # Remove the uploaded file
    os.remove(file_path)

    # Return as JSON string
    return json.dumps(answers)

# ----------------------------------------------------------------------------
# WebSocket endpoint
# ----------------------------------------------------------------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    clients[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        del clients[client_id]

# ----------------------------------------------------------------------------
# Image upload endpoint
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Background task to process queue
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Run queue processing as a background task
# ----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue_task())

# ----------------------------------------------------------------------------
# Result fetch endpoint
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Redirect root URL to index.html
# ----------------------------------------------------------------------------
@app.get("/")
async def read_root():
    return JSONResponse(
        {"message": "Redirecting to index.html"},
        headers={"Location": "/static/index.html"},
        status_code=307
    )
