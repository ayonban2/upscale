from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from queue import Queue
import shutil, os, uuid, subprocess, logging

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

task_queue = Queue()

# Ensure directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Real-ESRGAN API is running!"}

@app.post("/superres")
async def super_resolution(
    image: UploadFile = File(...),
    scale: int = Form(...)
):
    try:
        uid = str(uuid.uuid4())
        input_path = f"temp/{uid}_{image.filename}"
        output_path = f"temp/output_{uid}_{image.filename}"

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        process_task(input_path, output_path, scale)

        return FileResponse(output_path, media_type="image/png", filename=os.path.basename(output_path))
    
    except Exception as e:
        logging.error(f"Failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/create-payment")
def create_payment():
    return {"message": "Stripe integration coming soon"}

def process_task(input_path, output_path, scale):
    command = [
        "python", "realesrgan_manual_2pass_userinput.py",
        "--input", input_path,
        "--output", output_path,
        "--model", "models/RealESRGAN_x4plus.pth",
        "--scale", str(scale)
    ]
    subprocess.run(command, check=True)
