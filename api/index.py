import os, traceback, asyncio
import httpx
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
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

        import replicate
        output = await asyncio.to_thread(
            replicate.run,
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt[:1000],
                "num_inference_steps": 4,
                "num_outputs": 1,
                "width": 1080,
                "height": 1920,
            },
            api_token=REPLICATE_TOKEN,
        )

        if not output or not isinstance(output, list) or not output[0]:
            return JSONResponse(status_code=502, content={"error": "Replicate returned empty output"})

        img_url = output[0]
        async with httpx.AsyncClient(timeout=30.0) as client:
            img_resp = await client.get(img_url)
            img_resp.raise_for_status()

        return StreamingResponse(BytesIO(img_resp.content), media_type="image/jpeg")

    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
