from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil, uuid, os

app = FastAPI(title="Detector de Casas Colombianas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "models/house-yolo.pt"
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    return model

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    entrada = f"temp_{uuid.uuid4().hex}.jpg"
    salida  = f"static/resultado_{uuid.uuid4().hex}.jpg"
    try:
        with open(entrada, "wb") as f:
            shutil.copyfileobj(file.file, f)

        mdl = get_model()
        results = mdl(entrada)
        results[0].save(filename=salida)

        detecciones = []
        for box in results[0].boxes:
            detecciones.append({
                "confianza": round(float(box.conf[0]), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()]
            })

        return JSONResponse({
            "total_casas_detectadas": len(detecciones),
            "detecciones": detecciones,
            "imagen_resultado": "/" + salida.replace("\\", "/"),
            "mensaje": " Detección completada" if detecciones else " No se detectaron casas"
        })

    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(entrada):
            os.remove(entrada)

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": model is not None}

@app.post("/detectar-imagen")
async def detectar_imagen(file: UploadFile = File(...)):
    entrada = f"temp_{uuid.uuid4().hex}.jpg"
    salida  = f"static/resultado_{uuid.uuid4().hex}.jpg"
    try:
        with open(entrada, "wb") as f:
            shutil.copyfileobj(file.file, f)

        mdl = get_model()
        results = mdl(entrada)
        results[0].save(filename=salida)

        return FileResponse(salida, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(entrada):
            os.remove(entrada)

if __name__ == "__main__":
    import uvicorn
    print("\n API activa en http://localhost:8000")
    print(" Documentación en http://localhost:8000/docs\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)