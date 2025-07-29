import os
import base64
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Store reference images in memory
reference_images = {}

# Helper: encode image bytes to base64
def encode_image_bytes(image_bytes: bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

@app.post("/upload")
async def upload_image(
    testCase: str = Form(...),
    mode: str = Form(...),  # "reference" or "current"
    image: UploadFile = File(...)
):
    if not image.filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG images are allowed")

    image_bytes = await image.read()

    # ----------------- REFERENCE MODE -----------------
    if mode == "reference":
        reference_images[testCase] = image_bytes
        return JSONResponse({"success": True, "message": f"Reference saved for {testCase}"})

    # ----------------- CURRENT MODE -----------------
    elif mode == "current":
        if testCase not in reference_images:
            raise HTTPException(
                status_code=404,
                detail=f"No reference image found for test case '{testCase}'"
            )

        ref_bytes = reference_images[testCase]
        cur_bytes = image_bytes

        # Encode both images
        ref_b64 = encode_image_bytes(ref_bytes)
        cur_b64 = encode_image_bytes(cur_bytes)

        # GPT-4o prompt
        prompt = """
        Compare these two screenshots visually and describe:
        1. Similarities and differences
        2. Layout or UI changes
        3. Color or styling mismatches
        4. Missing/extra elements
        5. Whether they are visually equivalent
        Be detailed and precise.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}" }},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cur_b64}" }}
                        ]
                    }
                ],
                max_tokens=1200
            )

            comparison_text = response.choices[0].message.content

            return JSONResponse({
                "success": True,
                "testCase": testCase,
                "comparison": comparison_text
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Mode must be 'reference' or 'current'")
