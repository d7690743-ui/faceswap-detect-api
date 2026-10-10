import base64
import json
import os
import tempfile
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

app = FastAPI(title="FaceSwap Detection API", version="0.1.0")

# Set FRONTEND_ORIGIN to your exact frontend URL in Vercel.
# "*" is convenient for initial testing; restrict it before public use.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_face_app = None

def get_face_app():
    """Lazy-load Buffalo_L. First cold start can be slow and memory-heavy."""
    global _face_app
    if _face_app is None:
        try:
            _face_app = FaceAnalysis(
                name=os.getenv("INSIGHTFACE_MODEL", "buffalo_l"),
                providers=["CPUExecutionProvider"],
            )
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No se pudo cargar el modelo InsightFace. "
                    "Revisa los logs de Vercel y los límites de memoria/tiempo. "
                    f"Detalle: {type(exc).__name__}: {exc}"
                ),
            )
    return _face_app

@app.get("/")
def home():
    return {
        "service": "faceswap-detect-api",
        "status": "ok",
        "model": os.getenv("INSIGHTFACE_MODEL", "buffalo_l"),
        "endpoints": ["/health", "/detect"],
        "note": "El estado ok no significa que el modelo ya se haya descargado/cargado."
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_configured": os.getenv("INSIGHTFACE_MODEL", "buffalo_l")}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Sube un archivo de imagen.")

    # Protect the function from unexpectedly large uploads.
    max_bytes = 8 * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="La imagen debe pesar 8 MB o menos.")
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")

    face_app = get_face_app()
    faces = face_app.get(image)

    results: list[dict[str, Any]] = []
    for idx, face in enumerate(faces):
        bbox = [round(float(x), 2) for x in face.bbox.tolist()]
        result = {
            "index": idx,
            "bbox": bbox,
            "det_score": round(float(face.det_score), 4),
        }
        if getattr(face, "kps", None) is not None:
            result["landmarks"] = [
                [round(float(p[0]), 2), round(float(p[1]), 2)]
                for p in face.kps.tolist()
            ]
        # Return the embedding for use by a compatible swap pipeline.
        # This is biometric data: only send/store it with the person's consent.
        embedding = getattr(face, "normed_embedding", None)
        if embedding is not None:
            result["embedding"] = [round(float(x), 7) for x in embedding.tolist()]
        results.append(result)

    return {
        "filename": file.filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "face_count": len(results),
        "faces": results,
    }
