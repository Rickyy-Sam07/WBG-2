# Bengali Q&A Dataset Generator

A powerful, high-speed tool to automatically generate Bengali Question & Answer datasets from PDF documents or website URLs. Powered by **Groq + Llama 3.3 70B** for lightning-fast, high-quality Bengali language processing.

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed and get a **free API Key** from [Groq Console](https://console.groq.com/keys).

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/aswintguha/Bengali_QA.git
cd Bengali_QA

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```
GROQ_API_KEY_PRIMARY=your_first_groq_key
GROQ_API_KEY_SECONDARY=your_second_groq_key
```

When the app opens, it will show a required popup asking for both Groq keys. The keys are stored in your browser if you choose to remember them, and the generator will not run until both keys are entered.

### 4. Run the App
```bash
.venv/bin/python -m uvicorn app:app --reload --port 8001
```
Then open **[http://127.0.0.1:8001](http://127.0.0.1:8001)** in your browser.

> **Note for macOS users:** Modern Homebrew Python blocks global `pip install`. Always use the `.venv` virtual environment as shown above.

---

## Daily Usage Guide

### ▶️ Starting the App
Every time you want to open the app, run these two commands in Terminal:
```bash
cd bengali_qa
.venv/bin/python -m uvicorn app:app --reload --port 8001
```
Then open **http://127.0.0.1:8001** in your browser. The app is ready to use.

---

### ⏹️ Stopping the App
When you want to close the server and shut everything down:

1. Go to your **Terminal** where the server is running.
2. Press **`Ctrl + C`** — this immediately stops the server.
3. Close the Terminal window.

The browser tab will no longer load after this. Your generated CSV files are safely saved in the `output/` folder.

> **If you see `ERROR: Address already in use`** — a server is already running (possibly in another Terminal window). Run this to force-stop it:
> ```bash
> kill $(lsof -ti:8001)
> ```
> Then start the app again normally.

---

### 🔑 Changing Your Groq API Key
If you want to switch to a new API key:

1. Open the `.env` file in the `bengali_qa/` folder.
2. Replace the old key with your new one:
   ```
   GROQ_API_KEY=your_new_gsk_key_here
   ```
3. Save the file.
4. If the server is running, **stop it** (`Ctrl + C` in Terminal).
5. Start the server again:
   ```bash
   cd bengali_qa
   .venv/bin/python -m uvicorn app:app --reload --port 8001
   ```
The server always reads the `.env` file fresh on startup, so the new key will be active immediately.

---

## Features

- **Double Source Support**: Process local PDF files or any Bengali news/article URL.
- **Smart PDF Extraction**: Uses **PyMuPDF** for reliable text extraction from digital PDFs.
- **Dual Model Routing**: Uses `meta-llama/llama-4-scout-17b-16e-instruct` for 3 calls, then `moonshotai/kimi-k2-instruct-0905` for 3 calls.
- **Resilient Retry**: On any model error, automatically switches model, waits 10s, and retries up to 4 times.
- **High-Speed Generation**: Powered by Groq-hosted instruction models with automatic failover.
- **Batch Processing**: Automatically splits large documents into chunks (~1 page each).
- **CSV Export**: Downloads a clean dataset with `Question`, `Answer`, and `Category` (Factual, Causal, Reasoning).

---

## Project Structure

- `app.py`: FastAPI server logic & API integrations.
- `utils.py`: PDF processing, URL extraction, and text batching.
- `templates/`: Modern, responsive dashboard UI.
- `output/`: Automatically saves generated CSV datasets locally.
- `.env`: (Private) Stores your Groq API Key — never commit this to GitHub.
- `requirements.txt`: All Python dependencies.

---

## Tips for Best Results

- **PDF Quality**: Use text-based/digital PDFs. Scanned/image-only PDFs are not supported.
- **URL Support**: Works best with single articles from sites like Prothom Alo, Anandabazar, etc.
- **Scale**: Generates ~10 Q&A pairs per page for high data density.
- **Free Tier**: Groq allows 14,400 requests/day and 30 requests/minute — more than enough for large books.
