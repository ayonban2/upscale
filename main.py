import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import subprocess
import os
import uuid

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Real-ESRGAN API is running!"}

@app.post("/superres")
async def super_resolution(
    image: UploadFile = File(...),
    scale: int = Form(...),  # scale is sent as a form field
):
    # Unique filename handling
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{image.filename}"
    input_path = f"temp/{filename}"
    output_path = f"temp/output_{filename}"

    os.makedirs("temp", exist_ok=True)

    # Save uploaded image to temp folder
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Run Real-ESRGAN inference
    command = [
        "python",
        "realesrgan_manual_2pass_userinput.py",
        "--input", input_path,
        "--output", output_path,
        "--model", "models/RealESRGAN_x4plus.pth",
        "--scale", str(scale)
    ]
    subprocess.run(command, check=True)

    # Return output image as downloadable file
    return FileResponse(output_path, media_type="image/png", filename=os.path.basename(output_path))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
