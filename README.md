# 🎓 StudentWell AI — Emotion-Aware Student Support Assistant

An academic support chatbot that detects the emotional tone of student messages and adapts its responses accordingly.  
Built with **DistilBERT** (emotion classifier) + **Qwen2.5-3B-Instruct via Ollama** (response generator) + **Streamlit** (UI).

---

## 📋 Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ollama | Latest |
| macOS (Apple Silicon M4) | Recommended |
| RAM | 8 GB minimum |

---

## 🚀 Quick Start (Everything Already Set Up)

If the project is already set up and trained, just run:

```bash
# Step 1 — Make sure Ollama is running (in a separate terminal)
ollama serve

# Step 2 — Activate the virtual environment
cd StudentWell-AI-New
source .venv/bin/activate

# Step 3 — Launch the app
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 🔧 Fresh Setup (New Machine / VSCode)

### 1. Clone or copy the project

```bash
# If using git
git clone <your-repo-url>
cd StudentWell-AI-New

# Or just open the folder in VSCode / any editor
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

### 3. Start Ollama and pull the model

```bash
# Start Ollama server (run this in a dedicated terminal — keep it open)
ollama serve

# In a new terminal, pull the model (only needed once, ~1.9 GB)
ollama pull qwen2.5:3b

# Verify it downloaded
ollama list
```

You should see `qwen2.5:3b` in the list.

### 4. Create a Python virtual environment

```bash
cd StudentWell-AI-New

python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

> ✅ The trained DistilBERT model is already in `models/distilbert_emotion/`  
> ✅ The dataset is already in `data/`  
> ✅ No retraining needed unless you want to regenerate data

---

## 🧪 Testing Commands

### Check Ollama is running

```bash
# Check Ollama process
ollama list

# Or test qwen2.5:3b directly
ollama run qwen2.5:3b "Say hello"
```

### Check the emotion classifier works

```bash
source .venv/bin/activate
python src/model/predict_emotion.py
```

Expected output — it runs 5 test predictions and prints emotion + confidence for each.

### Check Ollama connection from the app

```bash
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0,'.')
from src.generation.llm_response import check_ollama_available
ok, msg = check_ollama_available()
print(msg)
"
```

### Run the full safety checker test suite

```bash
source .venv/bin/activate
python src/safety/safety_checker.py
```

Expected: 11/11 tests pass.

### Run an end-to-end generation test

```bash
source .venv/bin/activate
python src/generation/llm_response.py
```

This generates one emotion-aware response and one normal response and prints both.

---

## 🏋️ Retraining from Scratch (Optional)

Only needed if you want to regenerate the dataset or retrain the model.

```bash
source .venv/bin/activate

# Step 1 — Generate dataset (~2–3 hours, generates ~4000 samples)
python src/data/generate_dataset.py

# Step 2 — Validate and clean
python run_pipeline.py --step validate

# Step 3 — Preprocess and split
python run_pipeline.py --step preprocess

# Step 4 — Train DistilBERT (~15–20 min on M4)
python run_pipeline.py --step train

# Step 5 — Evaluate on test set
python run_pipeline.py --step evaluate

# Or run all steps at once (after generation is complete)
python run_pipeline.py
```

---

## 📂 Project Structure

```
StudentWell-AI-New/
│
├── app.py                          ← Streamlit entry point  (run this)
├── run_pipeline.py                 ← Training pipeline runner
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── generate_dataset.py     ← Generates student conversation dataset via Qwen
│   │   ├── validate_dataset.py     ← Cleans and validates the dataset
│   │   ├── preprocess.py           ← Tokenises and splits by conversation ID
│   │   └── test_generation_smoke.py
│   │
│   ├── model/
│   │   ├── train_classifier.py     ← Fine-tunes DistilBERT on MPS
│   │   ├── evaluate_classifier.py  ← Test-set metrics, confusion matrix
│   │   └── predict_emotion.py      ← Reusable prediction module (used by app)
│   │
│   ├── generation/
│   │   ├── emotion_policy.py       ← Maps emotion → response style rules
│   │   ├── prompt_builder.py       ← Builds system prompts for Qwen
│   │   └── llm_response.py         ← Calls Qwen via Ollama
│   │
│   ├── safety/
│   │   └── safety_checker.py       ← Rule-based safety layer (runs first)
│   │
│   ├── conversation/
│   │   └── conversation_manager.py ← Session history and context management
│   │
│   ├── evaluation/
│   │   ├── rating_store.py         ← Saves student ratings to SQLite
│   │   └── analytics.py            ← Chart generation for dashboard
│   │
│   └── pages/
│       ├── chat_page.py            ← 💬 Home/Chat page
│       ├── comparison_page.py      ← ⚖️  Normal vs Emotion-Aware
│       ├── evaluation_page.py      ← ⭐ Student Ratings
│       ├── dashboard_page.py       ← 📊 Analytics Dashboard
│       ├── model_performance_page.py ← 🎯 Model Results
│       └── about_page.py           ← ℹ️  About
│
├── data/
│   ├── raw/
│   │   ├── student_conversations.csv        ← Generated dataset (4,004 rows)
│   │   └── student_conversations_clean.csv  ← Cleaned dataset (3,990 rows)
│   └── processed/
│       ├── label_map.json          ← Emotion → integer mapping
│       ├── train.csv / val.csv / test.csv
│       └── train/ val/ test/       ← HuggingFace Arrow datasets
│
├── models/
│   └── distilbert_emotion/         ← Fine-tuned model weights + tokenizer
│
├── assets/
│   ├── training_loss_curve.png
│   ├── confusion_matrix.png
│   └── per_class_metrics.png
│
└── logs/
    ├── training.log
    ├── evaluation_report.txt
    └── dataset_generation.log
```

---

## 📊 Model Results

| Metric | Score |
|---|---|
| Test Accuracy | **79.3%** |
| Test Macro F1 | **0.7911** |
| Avg Confidence | 83.5% |

| Emotion | F1 |
|---|---|
| fear | 0.89 |
| surprise | 0.88 |
| disgust | 0.86 |
| sadness | 0.79 |
| joy | 0.76 |
| anger | 0.72 |
| neutral | 0.64 |

Trained on 2,799 samples, tested on 599 — split by conversation ID to prevent leakage.

---

## ⚠️ Known Limitations

- **Synthetic dataset** — training data was generated by Qwen2.5:3b. Labels are model-assigned, not human-verified.
- **Safety detection** — keyword/pattern-based. Cannot understand metaphor or sarcasm. Will over-trigger on "I want to die laughing". May miss indirect distress signals.
- **Not a mental health tool** — this is an academic support assistant only.
- **Session memory** — conversation history lives in browser session state and resets on close.
- **Ollama must be running** — the app shows an error if Ollama is not started before launch.
- **Neutral emotion** has the lowest F1 (0.64) — neutral messages are inherently ambiguous.

---

## 🌐 Moving to Another Machine or VSCode

1. Copy the entire `StudentWell-AI-New/` folder.
2. The `models/` folder contains trained weights — **do not delete it**.
3. The `data/` folder contains the dataset — keep it.
4. On the new machine: install Ollama, run `ollama pull qwen2.5:3b`, then follow **Fresh Setup** above.
5. The `.venv/` folder does **not** need to be copied — recreate it with `python3 -m venv .venv && pip install -r requirements.txt`.

---

## 💡 Ollama Tips

```bash
# Start Ollama server (must be running before the app)
ollama serve

# Check it is running
ollama list

# Check which port it uses (default 11434)
curl http://localhost:11434

# If Ollama is already running as a background service (macOS)
# it may start automatically — just check with ollama list
```

---

*StudentWell AI — built as an academic demonstration of emotion-aware GenAI for student support.*
