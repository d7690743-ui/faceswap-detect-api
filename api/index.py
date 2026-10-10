import os
import httpx
from fastapi import FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="FaceSwap Detection API Proxy", version="0.2.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def get_space_url():
    url = os.getenv("HF_SPACE_URL", "").strip().rstrip("/")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar HF_SPACE_URL en las variables de entorno de Vercel.",
        )
    return url

def get_headers():
    headers = {}
    key = os.getenv("INFERENCE_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers

@app.get("/")
def home():
    return {
        "service": "faceswap-detect-api",
        "status": "ok",
        "mode": "proxy",
        "inference_configured": bool(os.getenv("HF_SPACE_URL", "").strip()),
    }

@app.get("/health")
async def health():
    url = get_space_url()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url + "/health", headers=get_headers())
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar el servicio de inferencia: {type(exc).__name__}")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Sube un archivo de imagen.")
    content = await file.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="La imagen debe pesar 8 MB o menos.")
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    url = get_space_url()
    files = {
        "file": (
            file.filename or "image.jpg",
            content,
            file.content_type or "application/octet-stream",
        )
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                url + "/detect",
                files=files,
                headers=get_headers(),
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="El servicio de inferencia tardó demasiado. Espera a que despierte y vuelve a intentarlo.")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar el servicio de inferencia: {type(exc).__name__}")
