import os
import time
import csv
import re
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

from groq import Groq
try:
    from .utils import extract_text_from_pdf, extract_text_from_url, split_into_batches
except ImportError:
    from utils import extract_text_from_pdf, extract_text_from_url, split_into_batches

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "output"

# ── Load environment variables ──
project_env = BASE_DIR / ".env"
workspace_env = BASE_DIR.parent / ".env"
if project_env.exists():
    load_dotenv(dotenv_path=project_env)
elif workspace_env.exists():
    load_dotenv(dotenv_path=workspace_env)
else:
    load_dotenv()

# ── Dual model orchestration config ──
MODEL_NAME_PRIMARY = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_NAME_SECONDARY = "moonshotai/kimi-k2-instruct-0905"

CALLS_PER_MODEL = 3
CYCLE_LENGTH = CALLS_PER_MODEL * 2
MAX_RETRIES = 4
ERROR_SWITCH_WAIT_SECONDS = 10
CYCLE_WAIT_SECONDS = 3
LOW_TEMPERATURE = 0.2

# ── FastAPI app setup ──
app = FastAPI(title="Bengali Q&A Generator")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Fix for Python 3.14 + Jinja2 LRU cache incompatibility
_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
    cache_size=0,   # disable cache — avoids dict-as-key TypeError in Python 3.14
)
templates = Jinja2Templates(env=_jinja_env)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  THE BENGALI Q&A PROMPT
# ─────────────────────────────────────────────
BENGALI_PROMPT_TEMPLATE = """
তুমি একজন কঠোর সোর্স-গ্রাউন্ডেড বাংলা Q&A নির্মাতা।

নিচের বাংলা টেক্সট থেকে ১০ জোড়া প্রশ্ন-উত্তর তৈরি করো।
প্রতিটি প্রশ্ন-উত্তর নিচের ৩টি ভিন্ন ধরনের হতে হবে (সব মিলিয়ে ১০টি):
১. তথ্যমূলক (Factual) — সরাসরি তথ্য জিজ্ঞেস করে
২. কারণমূলক (Causal) — কেন/কীভাবে জিজ্ঞেস করে
৩. যুক্তিমূলক (Reasoning) — বিশ্লেষণ বা সিদ্ধান্ত জিজ্ঞেস করে

কঠোর নিয়ম:
1) প্রশ্ন ও উত্তর কেবল নিচের টেক্সট/URL কনটেন্ট থেকে হবে; বাইরের জ্ঞান, অনুমান বা কল্পনা ব্যবহার করা যাবে না।
2) উত্তর সংক্ষিপ্ত ও নির্ভুল হবে: সর্বোচ্চ ১ বাক্য, আদর্শভাবে ৮-২০ শব্দ।
3) টেক্সটে তথ্য না থাকলে সেটি নিয়ে প্রশ্ন-উত্তর তৈরি করবে না।
4) অতিরিক্ত ব্যাখ্যা, ভূমিকা বা উপসংহার লিখবে না।

উত্তর অবশ্যই বাংলায় দিতে হবে।

এই EXACT ফরম্যাটে লিখবে (কোনো অতিরিক্ত টেক্সট নয়):

Question: [বাংলায় প্রশ্ন]
Answer: [বাংলায় উত্তর]
Category: Factual

Question: [বাংলায় প্রশ্ন]
Answer: [বাংলায় উত্তর]
Category: Causal

Question: [বাংলায় প্রশ্ন]
Answer: [বাংলায় উত্তর]
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
    blocks = re.split(r'\n(?=(Question|Instruction):)', response_text.strip())

    for block in blocks:
        question_match = re.search(r'(?:Question|Instruction):\s*(.+?)(?=(?:Answer|Response):|$)', block, re.DOTALL)
        answer_match = re.search(r'(?:Answer|Response):\s*(.+?)(?=Category:|$)', block, re.DOTALL)
        category_match = re.search(r'Category:\s*(\w+)', block)

        if question_match and answer_match and category_match:
            rows.append({
                "Question": question_match.group(1).strip(),
                "Answer":    answer_match.group(1).strip(),
                "Category":    category_match.group(1).strip(),
            })

    return rows


def choose_model_for_call(call_count: int) -> str:
    cycle_position = call_count % CYCLE_LENGTH
    if cycle_position < CALLS_PER_MODEL:
        return MODEL_NAME_PRIMARY
    return MODEL_NAME_SECONDARY


def alternate_model(current_model: str) -> str:
    if current_model == MODEL_NAME_PRIMARY:
        return MODEL_NAME_SECONDARY
    return MODEL_NAME_PRIMARY


def run_with_retry(
    prompt: str,
    api_call_counter: dict,
    batch_number: int,
    total_batches: int,
    model_clients: dict[str, Groq],
) -> str:
    errors = []
    current_model = choose_model_for_call(api_call_counter["count"])
    max_attempts = MAX_RETRIES + 1

    for attempt in range(1, max_attempts + 1):
        call_number = api_call_counter["count"] + 1
        try:
            print(
                f"[app] Batch {batch_number}/{total_batches} | API call #{call_number} "
                f"| attempt {attempt}/{max_attempts} | model: {current_model}"
            )
            response = model_clients[current_model].chat.completions.create(
                model=current_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=LOW_TEMPERATURE,
                max_tokens=1400,
            )
            reply = response.choices[0].message.content

            if not reply or not reply.strip():
                raise ValueError("Model returned an empty response.")

            return reply

        except Exception as e:
            errors.append(f"attempt {attempt} ({current_model}): {e}")
            if attempt == max_attempts:
                raise RuntimeError("; ".join(errors))

            current_model = alternate_model(current_model)
            print(
                f"[app] ⚠️ Error occurred. Switching model and waiting "
                f"{ERROR_SWITCH_WAIT_SECONDS}s before retry..."
            )
            time.sleep(ERROR_SWITCH_WAIT_SECONDS)

        finally:
            api_call_counter["count"] += 1
            if api_call_counter["count"] % CYCLE_LENGTH == 0:
                print(
                    f"[app] Completed {CYCLE_LENGTH} API calls "
                    f"(3 + 3 cycle). Waiting {CYCLE_WAIT_SECONDS}s..."
                )
                time.sleep(CYCLE_WAIT_SECONDS)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    template = _jinja_env.get_template("index.html")
    content = template.render(request=request)
    return HTMLResponse(content=content)


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
    primary_api_key: str = Form(None),
    secondary_api_key: str = Form(None),
):
    all_rows = []
    source_name = "output"
    api_call_counter = {"count": 0}
    model_clients = None

    primary_api_key = (primary_api_key or "").strip()
    secondary_api_key = (secondary_api_key or "").strip()

    if not primary_api_key or not secondary_api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "Please enter both Groq API keys in the popup before generating content."},
        )

    model_clients = {
        MODEL_NAME_PRIMARY: Groq(api_key=primary_api_key),
        MODEL_NAME_SECONDARY: Groq(api_key=secondary_api_key),
    }

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

        batches = split_into_batches(text, chars_per_batch=2500)
        total_batches = len(batches)

        for i, batch_text in enumerate(batches):
            print(f"[app] Processing batch {i+1}/{total_batches}…")
            prompt = BENGALI_PROMPT_TEMPLATE.format(text=batch_text)

            try:
                reply = run_with_retry(
                    prompt,
                    api_call_counter,
                    i + 1,
                    total_batches,
                    model_clients,
                )
                parsed = parse_qa_response(reply)
                if not parsed:
                    raise ValueError("Response format was invalid; no Q&A rows were parsed.")
                all_rows.extend(parsed)
                print(f"[app] Batch {i+1}: got {len(parsed)} Q&A pairs ✓")
            except Exception as e:
                print(f"[app] ⚠️ Batch {i+1} failed: {e}")

        if not all_rows:
            return JSONResponse(
                status_code=500,
                content={"error": "Model returned no Q&A pairs. Try a different text."}
            )

        # Save CSV to output/ folder
        filename = f"{source_name}_bengali_qa.csv"
        filepath = OUTPUT_DIR / filename

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Question", "Answer", "Category"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[app] ✅ Done! Total Q&A pairs: {len(all_rows)} → saved to {filepath}")

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "total_pairs": len(all_rows),
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})
    finally:
        # Request-scoped cleanup: do not retain API keys/clients on server after request ends.
        if model_clients is not None:
            model_clients.clear()
        primary_api_key = None
        secondary_api_key = None


@app.get("/download/{filename}")
async def download(filename: str):
    """Serve a generated CSV file from the output/ folder."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="text/csv",
    )


@app.get("/health")
async def health():
    return {"status": "running"}
