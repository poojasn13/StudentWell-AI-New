"""
Streamlit Page — About / Project Information
"""

import streamlit as st


def render():
    st.title("ℹ️ About StudentWell AI")
    st.markdown("""
## Emotion-Aware Student Support Assistant

**StudentWell AI** is an academic support chatbot that detects the emotional tone of student messages and adapts its responses accordingly — providing more empathetic, contextually appropriate support than a standard chatbot.

---

### 🏗️ Architecture

| Component | Technology |
|---|---|
| Emotion Classifier | DistilBERT (fine-tuned, 7-class) |
| Language Model | Qwen2.5-3B-Instruct via Ollama |
| Framework | Python + Streamlit |
| Training | PyTorch + HuggingFace Transformers |
| Storage | SQLite (ratings) + Streamlit session state |

---

### 🎭 Emotion Labels

The classifier recognises 7 emotions:

| Label | Description |
|---|---|
| neutral | Calm, informational, matter-of-fact |
| joy | Happy, excited, celebratory |
| sadness | Sad, disappointed, discouraged |
| anger | Frustrated, angry, agitated |
| fear | Anxious, nervous, worried |
| surprise | Unexpected, startled, astonished |
| disgust | Repelled, strongly negative reaction |

---

### 🔄 How It Works

1. Student types a message.
2. **Safety check** (rule-based) runs first — if crisis/concern signals are detected, a safe response is returned immediately.
3. **DistilBERT** classifies the emotion from the message + conversation context.
4. The detected emotion selects a **response policy** (tone, empathy level, structure).
5. **Qwen2.5:3b** generates a response guided by the emotion policy and conversation history.
6. The response is displayed with optional emotion diagnostics.

---

### ⚠️ Limitations

- **Synthetic dataset:** The training dataset was generated using Qwen2.5:3b. Labels are model-assigned, not human-verified, unless explicitly reviewed.
- **Safety detection:** Uses keyword and pattern matching. Cannot understand metaphor, sarcasm, or indirect expressions. It may over-trigger on phrases like "I want to die laughing" and may miss indirect distress signals.
- **Not a mental health tool:** This system is designed for academic support. It is not a replacement for professional mental health services.
- **Local deployment:** The LLM runs locally via Ollama and requires the model to be pre-downloaded.
- **Session memory:** Conversation history is stored in browser session state and does not persist after closing.

---

### 📂 Project Structure

```
StudentWell-AI-New/
├── app.py                       ← Streamlit entry point
├── run_pipeline.py              ← Training pipeline runner
├── src/
│   ├── data/                    ← Dataset generation, validation, preprocessing
│   ├── model/                   ← DistilBERT training, evaluation, prediction
│   ├── generation/              ← Emotion policy, prompt builder, LLM generation
│   ├── safety/                  ← Safety checker
│   ├── conversation/            ← Conversation history manager
│   ├── evaluation/              ← Rating store, analytics
│   └── pages/                   ← Streamlit page modules
├── data/
│   ├── raw/                     ← Generated CSV dataset
│   └── processed/               ← Train/val/test splits + label map
├── models/
│   └── distilbert_emotion/      ← Fine-tuned model weights
├── assets/                      ← Charts and plots
└── logs/                        ← Training and generation logs
```

---

### 🚀 Setup & Run

```bash
# 1. Install Ollama and ensure qwen2.5:3b is available
ollama pull qwen2.5:3b
ollama serve

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Generate dataset (runs for several hours)
python src/data/generate_dataset.py

# 4. Train the classifier (after generation completes)
python run_pipeline.py

# 5. Launch the app
streamlit run app.py
```

---

*This is a student project built for educational purposes. It demonstrates the application of NLP, fine-tuned transformer models, and generative AI in a student support context.*
    """)
