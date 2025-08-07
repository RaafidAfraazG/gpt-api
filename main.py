# main.py
import os
import base64
import logging
from typing import Dict
from datetime import datetime, timedelta

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is required in .env file")
    raise RuntimeError("OPENAI_API_KEY is required")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(
    title="Image Comparison API",
    description="Simple image comparison using GPT-4 Vision",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_CACHE_SIZE = 50
CACHE_EXPIRY_HOURS = 24

# Storage
class ImageData:
    def __init__(self, data: bytes):
        self.data = data
        self.timestamp = datetime.now()

reference_images: Dict[str, ImageData] = {}

# Helper functions
def validate_file(file: UploadFile) -> bytes:
    """Validate and read file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed"
        )
    
    return file_ext

def cleanup_old_images():
    """Remove expired images"""
    current_time = datetime.now()
    expired = [
        key for key, img in reference_images.items()
        if current_time - img.timestamp > timedelta(hours=CACHE_EXPIRY_HOURS)
    ]
    
    for key in expired:
        del reference_images[key]
        logger.info(f"Removed expired image: {key}")

def manage_cache_size():
    """Keep cache size under limit"""
    if len(reference_images) > MAX_CACHE_SIZE:
        # Remove oldest images
        sorted_items = sorted(reference_images.items(), key=lambda x: x[1].timestamp)
        excess = len(reference_images) - MAX_CACHE_SIZE
        
        for i in range(excess):
            key = sorted_items[i][0]
            del reference_images[key]
            logger.info(f"Removed old image due to cache limit: {key}")

def encode_image(image_bytes: bytes) -> str:
    """Encode image to base64"""
    return base64.b64encode(image_bytes).decode("utf-8")

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Image Comparison API", "docs": "/docs"}

@app.get("/health")
async def health():
    """Health check"""
    try:
        # Test OpenAI connection
        client.models.list()
        return {
            "status": "healthy",
            "cached_images": len(reference_images),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/upload")
async def upload_image(
    testCase: str = Form(...),
    mode: str = Form(...),
    image: UploadFile = File(...)
):
    """Upload and process images - CORE FUNCTIONALITY UNCHANGED"""
    try:
        # Validate inputs
        if not testCase.strip():
            raise HTTPException(status_code=400, detail="testCase cannot be empty")
        
        if mode not in ["reference", "current"]:
            raise HTTPException(status_code=400, detail="Mode must be 'reference' or 'current'")
        
        # Validate file
        validate_file(image)
        
        # Read image
        image_bytes = await image.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Cleanup old images
        cleanup_old_images()
        manage_cache_size()
        
        # ----------------- REFERENCE MODE -----------------
        if mode == "reference":
            reference_images[testCase] = ImageData(image_bytes)
            logger.info(f"Reference saved: {testCase}")
            return JSONResponse({
                "success": True, 
                "message": f"Reference saved for {testCase}"
            })
        
        # ----------------- CURRENT MODE -----------------
        elif mode == "current":
            if testCase not in reference_images:
                raise HTTPException(
                    status_code=404,
                    detail=f"No reference image found for test case '{testCase}'"
                )
            
            # Get reference image
            ref_bytes = reference_images[testCase].data
            cur_bytes = image_bytes
            
            # Encode images
            ref_b64 = encode_image(ref_bytes)
            cur_b64 = encode_image(cur_bytes)
            
            # GPT-4 Vision comparison (UNCHANGED)
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
                logger.info(f"Starting comparison: {testCase}")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cur_b64}"}}
                            ]
                        }
                    ],
                    max_tokens=1200
                )
                
                comparison_text = response.choices[0].message.content
                logger.info(f"Comparison completed: {testCase}")
                
                return JSONResponse({
                    "success": True,
                    "testCase": testCase,
                    "comparison": comparison_text
                })
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Utility endpoints
@app.get("/references")
async def list_references():
    """List stored references"""
    refs = []
    for test_case, img_data in reference_images.items():
        refs.append({
            "testCase": test_case,
            "timestamp": img_data.timestamp.isoformat(),
            "age_hours": round((datetime.now() - img_data.timestamp).total_seconds() / 3600, 2)
        })
    return {"references": refs}

@app.delete("/references/{testCase}")
async def delete_reference(testCase: str):
    """Delete a reference"""
    if testCase in reference_images:
        del reference_images[testCase]
        return {"success": True, "message": f"Deleted {testCase}"}
    else:
        raise HTTPException(status_code=404, detail=f"Reference '{testCase}' not found")

@app.delete("/references")
async def clear_references():
    """Clear all references"""
    count = len(reference_images)
    reference_images.clear()
    return {"success": True, "message": f"Cleared {count} references"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )