import os
import re
import uuid
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

try:
    import assemblyai as aai
except Exception:
    aai = None


BASE_DIR = Path("/mnt/nas-brain/projects")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://host.docker.internal:6333")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384

GROQ_CHAT_MODEL = "llama-3.1-8b-instant"
GROQ_TRANSCRIBE_MODEL = "whisper-large-v3-turbo"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")

BASE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="NAS AI Brain Manager",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NAS AI Brain Manager")
st.write("Upload TXT/PDF/MP3, convert them to searchable text, update Qdrant, then ask questions about each project folder.")


def safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    name = name.strip("_")
    return name or "default_project"


def project_path(project: str) -> Path:
    return BASE_DIR / safe_name(project)


def collection_name(project: str) -> str:
    return f"brain_{safe_name(project)}"


def setup_project_dirs(project: str):
    root = project_path(project)
    folders = {
        "root": root,
        "uploads": root / "uploads",
        "txt": root / "uploads" / "txt",
        "pdf": root / "uploads" / "pdf",
        "mp3": root / "uploads" / "mp3",
        "text": root / "text",
        "transcripts": root / "transcripts",
        "processed": root / "processed",
        "logs": root / "logs",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_qdrant():
    return QdrantClient(url=QDRANT_URL)


@st.cache_resource
def load_local_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")


def chunk_text(text: str, chunk_size=1200, overlap=250):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


def transcribe_with_groq(mp3_file: Path):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing")

    size_mb = mp3_file.stat().st_size / (1024 * 1024)
    if size_mb > 24:
        raise RuntimeError(f"File too large for Groq direct upload: {size_mb:.2f} MB")

    client = Groq(api_key=GROQ_API_KEY)

    with open(mp3_file, "rb") as audio:
        result = client.audio.transcriptions.create(
            file=(mp3_file.name, audio.read()),
            model=GROQ_TRANSCRIBE_MODEL,
            language="ar",
            response_format="text",
            temperature=0,
        )

    return str(result), f"Groq {GROQ_TRANSCRIBE_MODEL}"


def transcribe_with_deepgram(mp3_file: Path):
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is missing")

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-3",
        "language": "ar",
        "smart_format": "true",
        "punctuate": "true",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mpeg",
    }

    with open(mp3_file, "rb") as audio:
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=audio,
            timeout=900,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"Deepgram HTTP {response.status_code}: {response.text[:1000]}")

    data = response.json()
    transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript, "Deepgram REST nova-3"


def transcribe_with_assemblyai(mp3_file: Path):
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY is missing")

    if aai is None:
        raise RuntimeError("AssemblyAI package is not available")

    aai.settings.api_key = ASSEMBLYAI_API_KEY

    config = aai.TranscriptionConfig(language_code="ar")
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(str(mp3_file))

    if transcript.status == "error":
        raise RuntimeError(transcript.error)

    return transcript.text, "AssemblyAI"


def transcribe_with_local(mp3_file: Path):
    model = load_local_whisper()

    segments, info = model.transcribe(
        str(mp3_file),
        language="ar",
        beam_size=5,
        vad_filter=True
    )

    lines = [
        f"Language: {info.language}",
        f"Duration: {info.duration}",
        ""
    ]

    for segment in segments:
        lines.append(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}")

    return "\n".join(lines), "Local faster-whisper base"


def transcribe_mp3(mp3_file: Path, out_file: Path):
    providers = [
        ("Groq", transcribe_with_groq),
        ("Deepgram", transcribe_with_deepgram),
        ("AssemblyAI", transcribe_with_assemblyai),
        ("Local", transcribe_with_local),
    ]

    errors = []

    for provider_name, provider_func in providers:
        try:
            transcript, provider_used = provider_func(mp3_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Source: {mp3_file}\n")
                f.write(f"Transcribed by: {provider_used}\n\n")
                f.write(transcript)

            return True, provider_used, errors

        except Exception as e:
            errors.append(f"{provider_name} failed: {e}")

    return False, None, errors


def convert_txt_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert_pdf_file(pdf_file: Path, txt_file: Path, processed_dir: Path):
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    code, out, err = run_cmd(["pdftotext", str(pdf_file), str(txt_file)])

    if txt_file.exists() and txt_file.stat().st_size > 300:
        return True, "pdftotext"

    ocr_pdf = processed_dir / f"{pdf_file.stem}_ocr.pdf"
    code, out, err = run_cmd([
        "ocrmypdf",
        "--language", "ara+eng",
        "--deskew",
        "--rotate-pages",
        "--skip-text",
        str(pdf_file),
        str(ocr_pdf)
    ])

    if code != 0:
        return False, f"OCR failed: {err[:500]}"

    code, out, err = run_cmd(["pdftotext", str(ocr_pdf), str(txt_file)])

    if txt_file.exists() and txt_file.stat().st_size > 0:
        return True, "ocrmypdf + pdftotext"

    return False, "No text extracted"


def rebuild_qdrant(project: str):
    folders = setup_project_dirs(project)
    qdrant = load_qdrant()
    embedder = load_embedder()
    coll = collection_name(project)

    collections = [c.name for c in qdrant.get_collections().collections]

    if coll in collections:
        qdrant.delete_collection(collection_name=coll)

    qdrant.create_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    text_sources = []
    text_sources.extend(list(folders["text"].rglob("*.txt")))
    text_sources.extend(list(folders["transcripts"].rglob("*.txt")))

    batch = []
    total_chunks = 0

    for txt_file in text_sources:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            vector = embedder.encode(chunk).tolist()

            batch.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "project": safe_name(project),
                        "source": str(txt_file),
                        "chunk": idx,
                        "text": chunk,
                    },
                )
            )

            total_chunks += 1

            if len(batch) >= 64:
                qdrant.upsert(collection_name=coll, points=batch)
                batch = []

    if batch:
        qdrant.upsert(collection_name=coll, points=batch)

    return coll, len(text_sources), total_chunks


def ask_project(project: str, question: str, limit: int):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing")

    qdrant = load_qdrant()
    embedder = load_embedder()
    groq_client = Groq(api_key=GROQ_API_KEY)
    coll = collection_name(project)

    query_vector = embedder.encode(question).tolist()

    query_result = qdrant.query_points(
        collection_name=coll,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    results = query_result.points

    context_parts = []
    for r in results:
        source = r.payload.get("source", "")
        text = r.payload.get("text", "")
        context_parts.append(f"Source: {source}\nText:\n{text}")

    context = "\n\n--------------------\n\n".join(context_parts)

    system_prompt = """
أنت مساعد عربي متخصص في الإجابة اعتماداً فقط على النصوص المسترجعة من قاعدة معرفة المستخدم.
لا تستخدم معلومات خارج النصوص.
إذا كانت النصوص غير كافية، قل ذلك بوضوح.
إذا كان النص فيه أخطاء تفريغ أو OCR، وضح أن المعنى مبني على النص المتاح.
"""

    user_prompt = f"""
السؤال:
{question}

النصوص المسترجعة:
{context}

المطلوب:
1. أجب بالعربية بشكل واضح.
2. ابدأ بعبارة: "بحسب النصوص المسترجعة..."
3. لا تضف معلومات من خارج النص.
4. أضف قسمًا بعنوان: "الدليل من النص" مع اقتباسين قصيرين.
5. في النهاية اذكر أسماء الملفات المستخدمة فقط.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=900,
    )

    return response.choices[0].message.content, results


existing_projects = sorted([p.name for p in BASE_DIR.iterdir() if p.is_dir()])

st.sidebar.header("Project")
new_project = st.sidebar.text_input(
    "Create/select project folder",
    value=existing_projects[0] if existing_projects else "elsharawy"
)

project = safe_name(new_project)
folders = setup_project_dirs(project)

st.sidebar.write("Current project:")
st.sidebar.code(project)

st.sidebar.write("Qdrant collection:")
st.sidebar.code(collection_name(project))

tab_upload, tab_convert, tab_index, tab_ask, tab_status = st.tabs([
    "1 Upload",
    "2 Convert / Transcribe / OCR",
    "3 Update Qdrant",
    "4 Ask",
    "5 Status"
])

with tab_upload:
    st.subheader("Upload files to this project")

    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, or MP3 files",
        type=["txt", "pdf", "mp3"],
        accept_multiple_files=True
    )

    if st.button("Save uploaded files"):
        if not uploaded_files:
            st.warning("No files uploaded.")
        else:
            for uploaded in uploaded_files:
                filename = Path(uploaded.name).name
                ext = Path(filename).suffix.lower()

                if ext == ".txt":
                    dst = folders["txt"] / filename
                elif ext == ".pdf":
                    dst = folders["pdf"] / filename
                elif ext == ".mp3":
                    dst = folders["mp3"] / filename
                else:
                    st.warning(f"Unsupported file skipped: {filename}")
                    continue

                with open(dst, "wb") as f:
                    f.write(uploaded.getbuffer())

                st.success(f"Saved: {dst}")

with tab_convert:
    st.subheader("Convert TXT/PDF/MP3 to searchable text")

    st.info("TXT is copied. PDF uses pdftotext then OCR. MP3 uses Groq → Deepgram → AssemblyAI → local fallback.")

    if st.button("Run conversion for this project"):
        logs = []

        for txt in folders["txt"].rglob("*.txt"):
            rel = txt.relative_to(folders["txt"])
            out = folders["text"] / rel
            if out.exists() and out.stat().st_size > 0:
                logs.append(f"TXT skipped existing: {out}")
                continue
            convert_txt_file(txt, out)
            logs.append(f"TXT copied: {txt} -> {out}")

        for pdf in folders["pdf"].rglob("*.pdf"):
            rel = pdf.relative_to(folders["pdf"])
            out = folders["text"] / rel.with_suffix(".txt")
            processed_dir = folders["processed"] / rel.parent

            if out.exists() and out.stat().st_size > 0:
                logs.append(f"PDF skipped existing: {out}")
                continue

            ok, method = convert_pdf_file(pdf, out, processed_dir)
            if ok:
                logs.append(f"PDF converted using {method}: {pdf}")
            else:
                logs.append(f"PDF FAILED: {pdf} | {method}")

        for mp3 in folders["mp3"].rglob("*.mp3"):
            rel = mp3.relative_to(folders["mp3"])
            out = folders["transcripts"] / rel.with_suffix(".txt")

            if out.exists() and out.stat().st_size > 0:
                logs.append(f"MP3 skipped existing: {out}")
                continue

            ok, provider, errors = transcribe_mp3(mp3, out)
            if ok:
                logs.append(f"MP3 transcribed using {provider}: {mp3}")
            else:
                logs.append(f"MP3 FAILED: {mp3}")
                logs.extend(errors)

        st.text_area("Conversion log", "\n".join(logs), height=500)

with tab_index:
    st.subheader("Update Qdrant collection")
    st.warning("This rebuilds the Qdrant collection for this project from converted text/transcripts.")

    if st.button("Rebuild Qdrant for this project"):
        with st.spinner("Indexing text into Qdrant..."):
            coll, file_count, chunk_count = rebuild_qdrant(project)

        st.success("Qdrant updated.")
        st.write(f"Collection: `{coll}`")
        st.write(f"Text/transcript files indexed: `{file_count}`")
        st.write(f"Chunks indexed: `{chunk_count}`")

with tab_ask:
    st.subheader("Ask a question about this project")

    question = st.text_area(
        "Question",
        placeholder="مثال: ماذا قال الشيخ الشعراوي عن قصة آدم؟",
        height=120
    )

    limit = st.slider("Number of retrieved chunks", 3, 12, 6)

    if st.button("Ask this project"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching Qdrant and asking Groq..."):
                try:
                    answer, results = ask_project(project, question, limit)
                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Retrieved Sources")
                    for r in results:
                        source = r.payload.get("source", "")
                        text = r.payload.get("text", "")
                        with st.expander(f"{Path(source).name} - score {r.score:.4f}"):
                            st.write(source)
                            st.text(text[:2500])

                except Exception as e:
                    st.error(str(e))

with tab_status:
    st.subheader("Project status")

    txt_count = len(list(folders["txt"].rglob("*.txt")))
    pdf_count = len(list(folders["pdf"].rglob("*.pdf")))
    mp3_count = len(list(folders["mp3"].rglob("*.mp3")))
    text_count = len(list(folders["text"].rglob("*.txt")))
    transcript_count = len(list(folders["transcripts"].rglob("*.txt")))

    st.write(f"Project path: `{folders['root']}`")
    st.write(f"Uploaded TXT: `{txt_count}`")
    st.write(f"Uploaded PDF: `{pdf_count}`")
    st.write(f"Uploaded MP3: `{mp3_count}`")
    st.write(f"Converted text files: `{text_count}`")
    st.write(f"Transcript files: `{transcript_count}`")

    st.write("API keys:")
    st.write(f"Groq key present: `{bool(GROQ_API_KEY)}`")
    st.write(f"Deepgram key present: `{bool(DEEPGRAM_API_KEY)}`")
    st.write(f"AssemblyAI key present: `{bool(ASSEMBLYAI_API_KEY)}`")

    st.write("Folders:")
    for name, path in folders.items():
        st.code(f"{name}: {path}")
