# 🤖 ANTIGRAVITY MASTER BLUEPRINT
## Bengali Q&A Dataset Generator — Full Build Instructions

> **To Antigravity Agent:** Read this entire file carefully. Then execute every step in order.
> Create all folders, all files, and all code exactly as written below.
> Do not skip any section. Do not ask for confirmation — just build it.

---

## STEP 0 — CREATE THE PROJECT STRUCTURE

Run these shell commands to set up the project:

```bash
mkdir -p bengali_qa/templates
mkdir -p bengali_qa/static
mkdir -p bengali_qa/output
cd bengali_qa
```

---

## STEP 1 — INSTALL ALL DEPENDENCIES

Run this in terminal inside the `bengali_qa/` folder:

```bash
brew install tesseract
brew install tesseract-lang
brew install poppler
pip install fastapi uvicorn google-generativeai pdfplumber \
    trafilatura pytesseract pandas python-multipart \
    pdf2image pillow python-dotenv
```

---

## STEP 2 — CREATE `.env` FILE

Create a file named `.env` inside `bengali_qa/` with this content.
The user will paste their own Gemini API key here.

```
GEMINI_API_KEY=paste_your_gemini_api_key_here
```

> ⚠️ The user must replace `paste_your_gemini_api_key_here` with their real key from https://aistudio.google.com/app/apikey

---

## STEP 3 — CREATE `utils.py`

Create a file named `utils.py` inside `bengali_qa/` with this EXACT content:

```python
import os
import io
import pytesseract
import pdfplumber
import trafilatura

from PIL import Image
from pdf2image import convert_from_bytes


def extract_text_from_url(url: str) -> str:
    """
    Downloads a webpage and extracts only the main article text.
    Works for Bengali news sites like Prothom Alo, Anandabazar, etc.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not download content from URL: {url}")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text or len(text.strip()) < 50:
        raise ValueError("No readable text found at this URL. Try a different link.")

    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Smart PDF reader:
      1. Tries pdfplumber first (fast, for digital PDFs).
      2. Falls back to Tesseract OCR with Bengali ('ben') language pack
         for image/scanned PDFs.
    """
    full_text_parts = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total_pages = len(pdf.pages)
        print(f"[utils] PDF has {total_pages} pages.")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and len(text.strip()) > 30:
                full_text_parts.append(text.strip())
                print(f"[utils] Page {i+1}: digital text ✓")
            else:
                print(f"[utils] Page {i+1}: image detected → running Bengali OCR…")
                images = convert_from_bytes(
                    file_bytes,
                    first_page=i + 1,
                    last_page=i + 1,
                    dpi=300,
                )
                if images:
                    ocr_text = pytesseract.image_to_string(
                        images[0],
                        lang="ben",
                        config="--psm 6",
                    )
                    if ocr_text.strip():
                        full_text_parts.append(ocr_text.strip())

    if not full_text_parts:
        raise ValueError(
            "Could not extract any text. "
            "Make sure Tesseract + Bengali pack is installed: "
            "brew install tesseract tesseract-lang"
        )

    return "\n\n".join(full_text_parts)


def split_into_batches(text: str, chars_per_batch: int = 8000) -> list:
    """
    Splits large text into smaller chunks (~10 pages each)
    so we can send them one-by-one to the Gemini API for free tier safety.
    """
    words = text.split()
    batches = []
    current_batch = []
    current_length = 0

    for word in words:
        current_batch.append(word)
        current_length += len(word) + 1

        if current_length >= chars_per_batch:
            batches.append(" ".join(current_batch))
            current_batch = []
            current_length = 0

    if current_batch:
        batches.append(" ".join(current_batch))

    print(f"[utils] Text split into {len(batches)} batches.")
    return batches
```

---

## STEP 4 — CREATE `app.py`

Create a file named `app.py` inside `bengali_qa/` with this EXACT content:

```python
import os
import io
import time
import csv
import re

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import google.generativeai as genai
from utils import extract_text_from_pdf, extract_text_from_url, split_into_batches

# ── Load environment variables ──
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Please add it to your .env file.")

# ── Gemini 3 Flash Preview (newest & smartest free model) ──
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3-0-flash-preview")

# ── FastAPI app setup ──
app = FastAPI(title="Bengali Q&A Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────
#  THE BENGALI Q&A PROMPT
# ─────────────────────────────────────────────
BENGALI_PROMPT_TEMPLATE = """
তুমি একজন বিশেষজ্ঞ বাংলা শিক্ষক এবং ডেটাসেট নির্মাতা।

নিচের বাংলা টেক্সট থেকে ঠিক ৩ জোড়া প্রশ্ন-উত্তর তৈরি করো।
প্রতিটি প্রশ্ন-উত্তর নিচের ৩টি ভিন্ন ধরনের হতে হবে:
১. তথ্যমূলক (Factual) — সরাসরি তথ্য জিজ্ঞেস করে
২. কারণমূলক (Causal) — কেন/কীভাবে জিজ্ঞেস করে
৩. যুক্তিমূলক (Reasoning) — বিশ্লেষণ বা সিদ্ধান্ত জিজ্ঞেস করে

উত্তর অবশ্যই বাংলায় দিতে হবে।

এই EXACT ফরম্যাটে লিখবে (কোনো অতিরিক্ত টেক্সট নয়):

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Factual

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Causal

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Reasoning

টেক্সট:
\"\"\"
{text}
\"\"\"
"""


# ─────────────────────────────────────────────
#  PARSE GEMINI OUTPUT INTO ROWS
# ─────────────────────────────────────────────
def parse_qa_response(response_text: str) -> list:
    rows = []
    blocks = re.split(r'\n(?=Instruction:)', response_text.strip())

    for block in blocks:
        instruction_match = re.search(r'Instruction:\s*(.+?)(?=Response:|$)', block, re.DOTALL)
        response_match    = re.search(r'Response:\s*(.+?)(?=Category:|$)', block, re.DOTALL)
        category_match    = re.search(r'Category:\s*(\w+)', block)

        if instruction_match and response_match and category_match:
            rows.append({
                "Instruction": instruction_match.group(1).strip(),
                "Response":    response_match.group(1).strip(),
                "Category":    category_match.group(1).strip(),
            })

    return rows


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
):
    all_rows = []
    source_name = "output"

    try:
        if file and file.filename:
            file_bytes = await file.read()
            print(f"[app] Received PDF: {file.filename} ({len(file_bytes)} bytes)")
            text = extract_text_from_pdf(file_bytes)
            source_name = file.filename.replace(".pdf", "")
        elif url and url.strip():
            print(f"[app] Received URL: {url.strip()}")
            text = extract_text_from_url(url.strip())
            source_name = "url_content"
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a PDF or enter a URL."}
            )

        print(f"[app] Total text length: {len(text)} characters")

        batches = split_into_batches(text, chars_per_batch=8000)
        total_batches = len(batches)

        for i, batch_text in enumerate(batches):
            print(f"[app] Processing batch {i+1}/{total_batches}…")
            prompt = BENGALI_PROMPT_TEMPLATE.format(text=batch_text)

            try:
                response = model.generate_content(prompt)
                parsed = parse_qa_response(response.text)
                all_rows.extend(parsed)
                print(f"[app] Batch {i+1}: got {len(parsed)} Q&A pairs ✓")
            except Exception as e:
                print(f"[app] ⚠️ Batch {i+1} failed: {e}")

            if i < total_batches - 1:
                time.sleep(5)

        if not all_rows:
            return JSONResponse(
                status_code=500,
                content={"error": "Gemini returned no Q&A pairs. Try a different text."}
            )

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["Instruction", "Response", "Category"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(all_rows)

        csv_bytes = output.getvalue().encode("utf-8-sig")
        filename = f"{source_name}_bengali_qa.csv"

        print(f"[app] ✅ Done! Total Q&A pairs: {len(all_rows)}")

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})


@app.get("/health")
async def health():
    return {"status": "running", "model": "gemini-3-0-flash-preview"}
```

---

## STEP 5 — CREATE `templates/index.html`

Create a file named `index.html` inside the `templates/` folder with this EXACT content:

```html
<!DOCTYPE html>
<html lang="bn">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>বাংলা Q&A জেনারেটর</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    body {
      font-family: 'Hind Siliguri', sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      min-height: 100vh;
      color: #eaeaea;
    }
    .main-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 20px;
      backdrop-filter: blur(10px);
      padding: 2.5rem;
      margin-top: 3rem;
      box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    h1 { font-weight: 700; font-size: 2rem; color: #e2b96f; }
    .subtitle { color: #a0aec0; font-size: 1rem; margin-bottom: 2rem; }
    .model-badge {
      display: inline-block;
      background: linear-gradient(135deg, #6c63ff, #48bb78);
      color: white;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      margin-bottom: 1.5rem;
      letter-spacing: 0.05em;
    }
    .section-label {
      font-weight: 600;
      color: #90cdf4;
      margin-bottom: 0.5rem;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .upload-zone {
      border: 2px dashed rgba(144,205,244,0.4);
      border-radius: 14px;
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s;
      background: rgba(144,205,244,0.05);
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: #90cdf4;
      background: rgba(144,205,244,0.1);
    }
    .upload-zone .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .upload-zone p { color: #a0aec0; margin: 0; font-size: 0.9rem; }
    .upload-zone .file-chosen { color: #68d391; font-weight: 600; margin-top: 0.5rem; }
    .or-divider {
      text-align: center; color: #718096; margin: 1.5rem 0; position: relative;
    }
    .or-divider::before, .or-divider::after {
      content: ''; position: absolute; top: 50%;
      width: 42%; height: 1px; background: rgba(255,255,255,0.1);
    }
    .or-divider::before { left: 0; }
    .or-divider::after { right: 0; }
    .form-control {
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.15);
      color: #eaeaea; border-radius: 10px; padding: 0.75rem 1rem;
    }
    .form-control:focus {
      background: rgba(255,255,255,0.1); border-color: #90cdf4;
      color: #eaeaea; box-shadow: 0 0 0 3px rgba(144,205,244,0.2);
    }
    .form-control::placeholder { color: #718096; }
    .btn-generate {
      background: linear-gradient(135deg, #e2b96f, #d69e2e);
      border: none; color: #1a1a2e; font-weight: 700;
      font-size: 1.1rem; padding: 0.85rem 2rem; border-radius: 12px;
      width: 100%; margin-top: 1.5rem; transition: all 0.3s;
    }
    .btn-generate:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(226,185,111,0.4); }
    .btn-generate:disabled { opacity: 0.6; transform: none; }
    .progress-section { display: none; margin-top: 1.5rem; }
    .progress { height: 10px; border-radius: 10px; background: rgba(255,255,255,0.1); }
    .progress-bar {
      background: linear-gradient(90deg, #e2b96f, #68d391);
      border-radius: 10px; transition: width 0.5s ease;
    }
    .status-text { color: #a0aec0; font-size: 0.9rem; margin-top: 0.75rem; }
    .result-section { display: none; margin-top: 1.5rem; }
    .result-box {
      background: rgba(104,211,145,0.1);
      border: 1px solid rgba(104,211,145,0.3);
      border-radius: 14px; padding: 1.5rem; text-align: center;
    }
    .result-box h4 { color: #68d391; font-weight: 700; margin: 0.5rem 0; }
    .result-box p { color: #a0aec0; font-size: 0.9rem; }
    .btn-download {
      background: linear-gradient(135deg, #48bb78, #38a169);
      border: none; color: white; font-weight: 700;
      padding: 0.75rem 2rem; border-radius: 10px; margin-top: 1rem;
      text-decoration: none; display: inline-block; transition: all 0.3s;
    }
    .btn-download:hover { transform: translateY(-2px); color: white; }
    .error-section { display: none; margin-top: 1.5rem; }
    .error-box {
      background: rgba(252,129,129,0.1);
      border: 1px solid rgba(252,129,129,0.3);
      border-radius: 14px; padding: 1.25rem; color: #fc8181;
    }
    .tip-box {
      background: rgba(144,205,244,0.07);
      border-left: 3px solid #90cdf4;
      border-radius: 0 10px 10px 0;
      padding: 0.9rem 1.2rem; margin-top: 2rem;
      font-size: 0.85rem; color: #a0aec0;
    }
    .tip-box strong { color: #90cdf4; }
    #fileInput { display: none; }
  </style>
</head>
<body>
<div class="container" style="max-width: 700px;">
  <div class="main-card">
    <div class="text-center mb-4">
      <div style="font-size: 3rem;">🇧🇩</div>
      <h1>বাংলা Q&A জেনারেটর</h1>
      <div class="model-badge">✨ Gemini 3 Flash Preview — Newest Model</div>
      <p class="subtitle">PDF বা ওয়েবসাইট থেকে স্বয়ংক্রিয়ভাবে প্রশ্ন-উত্তর তৈরি করুন</p>
    </div>

    <form id="qaForm">
      <div class="section-label">📄 PDF আপলোড করুন</div>
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <div class="icon">📚</div>
        <p>বাংলা PDF এখানে টেনে আনুন অথবা ক্লিক করুন</p>
        <p style="font-size:0.8rem;color:#4a5568;margin-top:0.3rem;">Image PDF / Scanned Book সমর্থিত</p>
        <div class="file-chosen" id="fileChosen" style="display:none;"></div>
      </div>
      <input type="file" id="fileInput" accept=".pdf" onchange="handleFileSelect(this)"/>

      <div class="or-divider">অথবা</div>

      <div class="section-label">🌐 ওয়েবসাইট URL দিন</div>
      <input type="url" id="urlInput" class="form-control"
             placeholder="https://www.prothomalo.com/article/..."/>

      <button type="submit" class="btn btn-generate" id="generateBtn">
        ⚡ Q&A তৈরি করুন
      </button>
    </form>

    <div class="progress-section" id="progressSection">
      <div class="section-label">⏳ প্রক্রিয়া চলছে…</div>
      <div class="progress">
        <div class="progress-bar progress-bar-striped progress-bar-animated"
             id="progressBar" style="width:100%"></div>
      </div>
      <div class="status-text" id="statusText">Gemini 3-কে পাঠানো হচ্ছে…</div>
    </div>

    <div class="result-section" id="resultSection">
      <div class="result-box">
        <div style="font-size:3rem;">✅</div>
        <h4>সফলভাবে তৈরি হয়েছে!</h4>
        <p id="resultMessage">আপনার Q&A ডেটাসেট রেডি।</p>
        <a href="#" class="btn-download" id="downloadBtn">⬇️ CSV ডাউনলোড করুন</a>
      </div>
    </div>

    <div class="error-section" id="errorSection">
      <div class="error-box">
        <strong>⚠️ ত্রুটি:</strong> <span id="errorMessage"></span>
      </div>
    </div>

    <div class="tip-box">
      <strong>💡 টিপস:</strong> বড় বই (২০০+ পাতা) প্রক্রিয়া করতে ৩-৫ মিনিট লাগতে পারে।
      CSV ফাইলটি সরাসরি Excel বা Numbers-এ খুলবে।
      প্রতিটি সারিতে <strong>Instruction, Response, Category</strong> কলাম থাকবে।
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
  let selectedFile = null, downloadUrl = null;

  const uploadZone = document.getElementById('uploadZone');
  uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f && f.type === 'application/pdf') setFile(f);
  });

  function handleFileSelect(input) { if (input.files[0]) setFile(input.files[0]); }

  function setFile(file) {
    selectedFile = file;
    const el = document.getElementById('fileChosen');
    el.textContent = `✓ ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
    el.style.display = 'block';
    document.getElementById('urlInput').value = '';
  }

  function showSection(id) {
    ['progressSection','resultSection','errorSection'].forEach(s => {
      document.getElementById(s).style.display = s === id ? 'block' : 'none';
    });
  }

  document.getElementById('qaForm').addEventListener('submit', async e => {
    e.preventDefault();
    const url = document.getElementById('urlInput').value.trim();
    const btn = document.getElementById('generateBtn');
    if (!selectedFile && !url) { alert('PDF আপলোড করুন অথবা URL দিন।'); return; }

    if (downloadUrl) { URL.revokeObjectURL(downloadUrl); downloadUrl = null; }
    btn.disabled = true; btn.textContent = '⏳ প্রক্রিয়া চলছে…';
    showSection('progressSection');

    const statuses = [
      'টেক্সট পড়া হচ্ছে…','ব্যাচে ভাগ করা হচ্ছে…',
      'Gemini 3-কে পাঠানো হচ্ছে…','বাংলা Q&A তৈরি হচ্ছে…','CSV তৈরি হচ্ছে…'
    ];
    let si = 0;
    const iv = setInterval(() => {
      document.getElementById('statusText').textContent = statuses[si++ % statuses.length];
    }, 4000);

    try {
      const fd = new FormData();
      if (selectedFile) { fd.append('file', selectedFile); }
      else { fd.append('url', url); fd.append('file', new Blob([]), ''); }

      const res = await fetch('/generate', { method: 'POST', body: fd });
      clearInterval(iv);

      if (!res.ok) {
        const err = await res.json();
        document.getElementById('errorMessage').textContent = err.error || 'অজানা ত্রুটি।';
        showSection('errorSection'); return;
      }

      const blob = await res.blob();
      downloadUrl = URL.createObjectURL(blob);
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename=(.+)/);
      const filename = match ? match[1] : 'bengali_qa.csv';

      document.getElementById('downloadBtn').href = downloadUrl;
      document.getElementById('downloadBtn').download = filename;
      document.getElementById('resultMessage').textContent =
        `"${filename}" রেডি। নিচের বোতামে ক্লিক করুন।`;
      showSection('resultSection');
    } catch(err) {
      clearInterval(iv);
      document.getElementById('errorMessage').textContent = 'সংযোগ ব্যর্থ: ' + err.message;
      showSection('errorSection');
    } finally {
      btn.disabled = false; btn.textContent = '⚡ Q&A তৈরি করুন';
    }
  });
</script>
</body>
</html>
```

---

## STEP 6 — CREATE `test_gemini.py`

Create a file named `test_gemini.py` inside `bengali_qa/` to verify the API key before running the full app:

```python
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "paste_your_gemini_api_key_here":
    print("❌ ERROR: Please paste your real Gemini API key in the .env file first!")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3-0-flash-preview")

print("⏳ Testing Gemini 3 Flash Preview…")
response = model.generate_content(
    "এই বাংলা টেক্সট থেকে ১টি প্রশ্ন-উত্তর তৈরি করো: বাংলাদেশের রাজধানী ঢাকা।"
)
print("✅ SUCCESS! Gemini replied:")
print(response.text)
```

---

## STEP 7 — RUN ORDER (Do These In Order!)

```bash
# 1. Go into the project folder
cd bengali_qa

# 2. Test your API key first
python test_gemini.py

# 3. Only if test shows ✅ SUCCESS — start the full app
uvicorn app:app --reload --port 8001
```

Then open browser at: **http://127.0.0.1:8001**

---

## FINAL FILE STRUCTURE

```
bengali_qa/
├── .env                  ← Your Gemini API key
├── app.py                ← FastAPI + Gemini 3 Flash logic
├── utils.py              ← PDF reader + URL reader + batcher
├── test_gemini.py        ← Run this FIRST to verify key
├── templates/
│   └── index.html        ← Bengali dashboard UI
├── static/               ← (empty)
└── output/               ← (empty, CSVs go here)
```

---

## TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY not found` | Edit `.env` and paste your real key |
| `model not found` error | Model string must be exactly `gemini-3-0-flash-preview` |
| `tesseract not found` | `brew install tesseract tesseract-lang` |
| `poppler not found` | `brew install poppler` |
| Bengali garbled in Excel | Open with File → Import → UTF-8 encoding |
| API quota error | Wait 60 sec — free tier is 10 req/min for Gemini 3 |

---

*Blueprint v2.0 — Upgraded to Gemini 3 Flash Preview — MacBook Air + Antigravity*
