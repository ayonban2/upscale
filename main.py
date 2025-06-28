import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import subprocess

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Real-ESRGAN is running!"}

@app.post("/superres")
async def super_resolution(image: UploadFile = File(...)):
    input_path = f"input_{image.filename}"
    output_path = f"output_{image.filename}"

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Run Real-ESRGAN inference
    # Assumes the .pth model is in models/ and the script uses that path
    command = [
        "python",
        "realesrgan_manual_2pass_userinput.py",
        "--input", input_path,
        "--output", output_path,
        "--model", "models/RealESRGAN_x4plus.pth"
    ]
    subprocess.run(command, check=True)

    return FileResponse(output_path, media_type="image/png", filename=output_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
