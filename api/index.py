import os, traceback, asyncio
import httpx
import base64
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
MAX_RETRIES = 3

GEMINI_PROMPT = """You are an AI image analyzer. Describe this image in detail for an AI image generator.
Rules:
- If there is any text, logo, or branding related to "Поліс ЮЕй" — do NOT include it in the description.
- Replace branded elements with neutral alternatives (e.g., "a modern insurance company logo" instead of the actual logo).
- Focus on: composition, colors, people (pose, clothing, expression), objects, background, lighting, text (except banned).
- Return ONLY the description — no greetings, no explanations, no extra words."""

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

        async with httpx.AsyncClient(timeout=30.0) as client:
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

        prompt_encoded = quote(prompt[:500])
        gen_url = f"{POLLINATIONS_BASE}/{prompt_encoded}?width=1080&height=1920&nofeed=true"

        async with httpx.AsyncClient(timeout=60.0) as client:
            gen_resp = await client.get(gen_url)
            gen_resp.raise_for_status()

        return StreamingResponse(BytesIO(gen_resp.content), media_type=gen_resp.headers.get("content-type", "image/png"))

    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=e.response.status_code, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
