# Image Comparison API

Simple, production-ready FastAPI service for comparing images using GPT-4 Vision.

---

## Quick Start

### 1. Setup

```bash
# Create environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_actual_openai_api_key_here

# Install dependencies
pip install -r requirements.txt

# Run directly
python main.py

API Endpoints
POST /upload - Upload and compare images

GET /health - Health check

GET /references - List stored references

DELETE /references/{testCase} - Delete specific reference

DELETE /references - Clear all references

GET /docs - API documentation


Configuration

File Limits
Max file size: 10MB

Allowed formats: PNG, JPG, JPEG

Max cached images: 50

Cache expiry: 24 hours

Environment Variables
OPENAI_API_KEY (required) - Your OpenAI API key

PORT - Server port, default is 5000


