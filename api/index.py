import os, traceback, asyncio
import httpx
import base64
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

HF_SPACE_URL = "https://alex2026daaaa-flux-schnell-clean.hf.space/api/predict"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
MAX_RETRIES = 3

GEMINI_PROMPT = """You are an expert AI image analyst. Your task is to describe this image in EXTREME DETAIL so another AI can recreate it as closely as possible.

REQUIRED in your description:
- Read ALL text visible in the image (except anything containing "Поліс ЮЕй") — include the text content, font style, color, size, position
- Describe the exact composition: where elements are placed (left/right/center/top/bottom)
- People: gender, approximate age, clothing (colors, style), pose, facial expression, hairstyle, accessories
- Objects: every visible object, its shape, color, size, position
- Background: colors, gradients, patterns, whether photo/illustration/3D
- Colors: exact or approximate color scheme, dominant colors
- Lighting and mood: bright/dim, warm/cold, professional/casual
- Style: photo, 3D render, flat design, illustration, vector — be specific
- If there is a logo or branding, describe its shape and colors but replace it with a generic neutral version

RULES:
- If text or branding says "Поліс ЮЕй" — OMIT it entirely, replace with a generic insurance company name or logo
- Do NOT add, invent, or imagine extra elements
- Return ONLY the description — pure text, no greetings, no explanations, no markdown"""

@app.post("/")
async def clean_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"

        request_body = {
            "contents": [{
                "parts": [
                    {"text": GEMINI_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}}
                ]
            }]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(MAX_RETRIES):
                resp = await client.post(
                    f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                    json=request_body,
                )
                if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt * 5)
                    continue
                resp.raise_for_status()
                break
            result = resp.json()

        try:
            prompt = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return JSONResponse(status_code=502, content={"error": "Gemini returned unexpected response"})

        # Try HF Space (Gradio API), fallback to Inference API
        image_bytes = None
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Try Space
            try:
                space_resp = await client.post(
                    HF_SPACE_URL,
                    json={"data": [prompt[:1500]]},
                    headers={"Content-Type": "application/json"},
                )
                if space_resp.status_code == 200:
                    data = space_resp.json()
                    img_url = data["data"][0]["url"]
                    if img_url.startswith("data:image"):
                        b64 = img_url.split(",", 1)[1]
                        image_bytes = base64.b64decode(b64)
            except Exception:
                pass

            # Fallback to Inference API
            if image_bytes is None and HF_TOKEN:
                infer_resp = await client.post(
                    HF_INFERENCE_URL,
                    json={"inputs": prompt[:1500]},
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                )
                if infer_resp.status_code == 200:
                    image_bytes = infer_resp.content

            if image_bytes is None:
                return JSONResponse(status_code=502, content={"error": "Image generation failed (both Space and Inference API)"})

        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")

    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
