# NAS AI Brain

A Streamlit app for building a local AI knowledge brain from TXT, PDF, and MP3 files.

## Features

- Upload TXT, PDF, and MP3 files by project/folder
- Convert TXT files to searchable text
- Extract PDF text with pdftotext
- OCR scanned PDFs with ocrmypdf and Arabic/English Tesseract
- Transcribe MP3 files using provider fallback:
  - Groq
  - Deepgram
  - AssemblyAI
  - Local faster-whisper
- Index text into Qdrant
- Ask questions using Groq chat model
- Show retrieved source chunks and evidence

## Folder structure

The app expects NAS data at:

    /mnt/nas-brain

Project files are stored under:

    /mnt/nas-brain/projects/<project-name>

## Environment variables

Create .env from .env.example:

    cp .env.example .env
    nano .env

Add your API keys:

    GROQ_API_KEY=your_groq_key_here
    DEEPGRAM_API_KEY=your_deepgram_key_here
    ASSEMBLYAI_API_KEY=your_assemblyai_key_here
    QDRANT_URL=http://host.docker.internal:6333

## Run locally

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py --server.port 8098 --server.address 0.0.0.0

Open:

    http://SERVER_IP:8098

## Run with Docker

    docker build -t nas-ai-brain:latest .

    docker run -d \
      --name nas-ai-brain \
      --restart unless-stopped \
      -p 8098:8098 \
      --env-file .env \
      -v /mnt/nas-brain:/mnt/nas-brain \
      --add-host=host.docker.internal:host-gateway \
      nas-ai-brain:latest

## Qdrant

Qdrant should be running and reachable from the app.

Example:

    docker run -d \
      --name nas-ai-qdrant \
      --restart unless-stopped \
      -p 6333:6333 \
      -p 6334:6334 \
      -v /mnt/webapps/nas-ai-brain/qdrant_storage:/qdrant/storage \
      qdrant/qdrant:latest

## Notes

Do not commit .env or API keys.
