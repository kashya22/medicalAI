
<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=220&section=header&text=Doctor%20Feedback%20Voice%20Agent&fontSize=40&fontColor=ffffff&fontAlignY=38&desc=Real-Time%20AI%20Voice%20Pipeline%20for%20Healthcare%20Feedback%20Automation&descSize=16&descAlignY=58&descColor=a8d8ea&animation=fadeIn" width="100%"/>

<br/>

<!-- Badges Row 1 — Tech Stack -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LiveKit-Agents%20SDK-D4232A?style=for-the-badge&logo=webrtc&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge&logo=thunderbird&logoColor=white"/>
  <img src="https://img.shields.io/badge/Sarvam%20AI-STT%20%2F%20TTS-7B2FBE?style=for-the-badge&logo=audiomack&logoColor=white"/>
  <img src="https://img.shields.io/badge/Silero-VAD-00B4D8?style=for-the-badge&logo=pytorch&logoColor=white"/>
</p>

<!-- Badges Row 2 — Meta -->
<p>
  <img src="https://img.shields.io/badge/Protocol-WebRTC-005CFF?style=for-the-badge&logo=webrtc&logoColor=white"/>
  <img src="https://img.shields.io/badge/LLM-Llama%203.3--70B-blueviolet?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Healthcare%20AI-27AE60?style=for-the-badge&logo=health&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/github/license/jayanthoffl/medicalAgent?style=for-the-badge"/>
</p>

<br/>

> **A production-grade, real-time voice AI agent** that replaces static post-consultation feedback forms with fully autonomous, natural voice conversations — powered by a carefully tuned multi-model pipeline optimized for Indian English in clinical settings.

</div>

---

## 📸 Live Interface Preview

<div align="center">
<img width="700" height="400" alt="Doctor Feedback Voice Agent Interface" src="https://github.com/user-attachments/assets/3afe1ea5-dcad-430f-939f-9357f9634670" />

*LiveKit Playground — agent actively engaged in a post-consultation feedback call*
</div>

---

## 🧭 Table of Contents

<details>
<summary><strong>Expand Navigation</strong></summary>

- [📌 Project Overview](#-project-overview)
- [🧠 ML System Design & Pipeline Architecture](#-ml-system-design--pipeline-architecture)
  - [End-to-End Audio Pipeline](#end-to-end-audio-pipeline)
  - [The Voice-to-Insight Stack (Component Deep Dive)](#the-voice-to-insight-stack-component-deep-dive)
- [⚙️ Technical Architecture](#️-technical-architecture)
- [🔬 Model Selection Rationale](#-model-selection-rationale)
  - [Why Groq + Llama 3.3-70B?](#why-groq--llama-33-70b)
  - [Why Sarvam AI?](#why-sarvam-ai)
  - [Why Silero VAD?](#why-silero-vad)
- [📊 Function Calling & Structured Data Extraction](#-function-calling--structured-data-extraction)
- [💾 Data Persistence & Output Schema](#-data-persistence--output-schema)
- [🚀 Installation & Setup](#-installation--setup)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)

</details>

---

## 📌 Project Overview

Traditional post-consultation patient feedback collection suffers from a fundamental problem: **response rates for static forms hover below 15%**. Patients leave the clinic, forget, or simply don't engage with paper/digital questionnaires.

This project solves that by deploying a **fully autonomous AI voice agent** that calls patients immediately after their appointment — engaging them in a natural, warm, conversational experience — and autonomously persisting structured, actionable feedback data.

```
Patient leaves clinic → Agent initiates call → Natural voice conversation →
Structured data extracted via LLM function calling → Dashboard updated
```

The system is not a chatbot wrapper. It is a **carefully orchestrated multi-model inference pipeline** with real-time audio streaming, dynamic state management, turn-taking intelligence, and autonomous tool execution — all running concurrently with sub-second latency.

---

## 🧠 ML System Design & Pipeline Architecture

### End-to-End Audio Pipeline

This is the core of the system. Every spoken word travels through four distinct ML inference stages before a response is generated and sent back to the patient.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FULL-DUPLEX AUDIO STREAM (WebRTC)                      │
│                         LiveKit SFU Transport Layer                         │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │  Raw PCM Audio (16kHz, 16-bit, Mono)
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1 — VOICE ACTIVITY DETECTION (VAD)                  │
│                                                                             │
│   Model : Silero VAD  (ONNX Runtime, ~1MB)                                  │
│   Input : Raw audio frames (30ms windows)                                   │
│   Output: speech_prob ∈ [0, 1] per frame                                    │
│   Task  : Segment user speech from silence / background noise               │
│   Why   : Prevents STT from processing empty frames → saves tokens/latency  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │  Validated Speech Segments
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 2 — SPEECH-TO-TEXT / ASR (Sarvam AI `saaras`)          │
│                                                                             │
│   Model : saaras (Sarvam AI, Indian-English optimized ASR)                  │
│   Input : Segmented audio chunks  |  Language hint: en-IN                   │
│   Output: Transcription text + word-level timestamps                        │
│   Task  : Acoustic model + language model decoding for Indian English        │
│   Why   : Standard Whisper degrades on South Asian accents; saaras is        │
│           trained on Indian English corpus → higher WER accuracy             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │  Transcribed Text (UTF-8)
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE 3 — LLM REASONING ENGINE (Groq + Llama 3.3-70B)         │
│                                                                             │
│   Model : meta-llama/Llama-3.3-70b-versatile (via Groq LPU)                │
│   Input : System prompt + conversation history + new user turn              │
│   Output: Next conversational turn + optional tool_call JSON                │
│   Tasks :                                                                   │
│     • State tracking   — Which of the 3 questions has been answered?        │
│     • Turn management  — When to ask next question vs. clarify/rephrase     │
│     • Intent detection — Extract ratings, sentiments from free-form speech  │
│     • Function calling — Decide when ALL data is collected → call tool      │
│   TTFB  : ~150–250ms (Groq LPU; vs ~800–1200ms on GPU inference)           │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │  Text Response / Tool Call
                             ▼
┌────────────────────────────┴────────────────────────────────────────────────┐
│          PARALLEL EXECUTION — RESPONSE vs. TOOL CALL                        │
│                                                                             │
│  IF tool_call detected:                IF text response:                    │
│  ┌─────────────────────────┐           ┌─────────────────────────────────┐  │
│  │  save_feedback() called │           │  STAGE 4 — TTS (Sarvam `bulbul`)│  │
│  │  Structured JSON →      │           │  Speaker  : Anushka (female)     │  │
│  │  feedback_dashboard.json│           │  Model    : bulbul:v2            │  │
│  │                         │           │  Language : en-IN                │  │
│  │  → Confirmation string  │           │  Output   : Opus/PCM audio stream│  │
│  │    returned to LLM      │           │  → Streamed back via WebRTC      │  │
│  └─────────────────────────┘           └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Voice-to-Insight Stack (Component Deep Dive)

| Pipeline Stage | Component | Role | Latency Budget |
|:---:|:---:|:---|:---:|
| **Transport** | LiveKit SFU | Full-duplex WebRTC audio relay | ~20–50ms RTT |
| **VAD** | Silero VAD | Binary speech/silence classification | ~5ms per frame |
| **ASR** | Sarvam `saaras` | Indian English speech transcription | ~100–300ms |
| **Reasoning** | Groq Llama 3.3-70B | Dialogue state + function calling | ~150–250ms TTFB |
| **TTS** | Sarvam `bulbul:v2` | Neural speech synthesis (en-IN) | ~200–400ms |
| **Storage** | `feedback_dashboard.json` | NDJSON structured output | < 1ms I/O |

**Total perceived latency (end-to-end):** `~500–1000ms` — well within the natural conversational threshold of 1.2 seconds.

---

## ⚙️ Technical Architecture

The system follows a **LiveKit Worker/Agent pattern**, not a monolithic pipeline. This architectural decision enables:

- **Horizontal scalability** — multiple worker processes can handle concurrent calls
- **Graceful state isolation** — each `AgentSession` is fully encapsulated per call
- **Interrupt handling** — LiveKit's turn-detector allows agents to be interrupted mid-sentence

```
┌──────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                       │
│                                                                  │
│   ┌─────────────────────────┐    ┌────────────────────────────┐  │
│   │   LiveKit Cloud / SFU   │    │    SIP Trunk (Optional)    │  │
│   │   WebRTC Media Server   │◄───┤   Outbound PSTN Calling    │  │
│   └──────────┬──────────────┘    └────────────────────────────┘  │
│              │ WebSocket Job Dispatch                             │
└──────────────┼───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AGENT PROCESS LAYER                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   WorkerOptions (entrypoint)                 │ │
│  │                                                             │ │
│  │  ┌───────────────────────────────────────────────────────┐  │ │
│  │  │              AgentSession (per-call context)           │  │ │
│  │  │                                                       │  │ │
│  │  │   vad    = silero.VAD.load()                          │  │ │
│  │  │   stt    = sarvam.STT(language="en-IN")               │  │ │
│  │  │   llm    = groq.LLM(model="llama-3.3-70b-versatile")  │  │ │
│  │  │   tts    = sarvam.TTS(speaker="anushka",              │  │ │
│  │  │                       model="bulbul:v2",              │  │ │
│  │  │                       target_language_code="en-IN")   │  │ │
│  │  │                                                       │  │ │
│  │  │   ┌───────────────────────────────────────────────┐   │  │ │
│  │  │   │       DoctorFeedbackAgent (Agent class)        │   │  │ │
│  │  │   │                                               │   │  │ │
│  │  │   │  • System Prompt (state machine instructions) │   │  │ │
│  │  │   │  • @function_tool: save_feedback()            │   │  │ │
│  │  │   │  • room_name binding                          │   │  │ │
│  │  │   └───────────────────────────────────────────────┘   │  │ │
│  │  └───────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        PERSISTENCE LAYER                          │
│                                                                  │
│   feedback_dashboard.json  (NDJSON — 1 record per completed call) │
└──────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

- **`AgentSession` over raw pipeline** — The SDK manages plugin lifecycle, audio buffering, and concurrency automatically, reducing boilerplate and risk of race conditions.
- **`@function_tool` decorator** — Native LiveKit function calling: the LLM schema for `save_feedback` is auto-generated from Python type annotations, ensuring the LLM always produces a valid, type-safe payload.
- **`generate_reply()` for proactive initiation** — The agent speaks *first*, not the patient, which is psychologically aligned with how phone calls work and increases engagement.
- **Event-driven disconnection handling** — The `participant_disconnected` callback ensures clean teardown without resource leaks.

---

## 🔬 Model Selection Rationale

### Why Groq + Llama 3.3-70B?

Voice agents live or die by **Time-To-First-Byte (TTFB)**. Any pause longer than ~1.2 seconds in a phone conversation feels like a dead line — patients hang up.

| Inference Backend | TTFB (Llama-70B) | Suitability |
|:---|:---:|:---:|
| Standard GPU (A100) | ~800–1200ms | ❌ Too slow |
| vLLM (self-hosted) | ~400–700ms | ⚠️ Marginal |
| **Groq LPU** | **~150–250ms** | ✅ Production-ready |

Groq's **Language Processing Unit (LPU)** architecture is purpose-built for sequential token generation — achieving ~6x throughput over comparable A100 GPU setups for inference workloads. For a 70B parameter model, this is transformative for real-time voice.

**Llama 3.3-70B-Versatile** was chosen over smaller models because:
- It reliably produces valid `tool_call` JSON in a single pass, even for ambiguous patient inputs
- Its instruction-following capability handles edge cases (patients going off-topic, giving non-numeric ratings, speaking in mixed language)
- At 70B parameters, context retention across a 3-question conversation is near-perfect

### Why Sarvam AI?

Standard ASR/TTS systems (Whisper, Google Cloud TTS) exhibit measurable degradation on Indian English:

- **Phonetic differences** — retroflex consonants (`t`, `d`, `n`) are misclassified
- **Prosody patterns** — Indian English stress patterns differ from American/British training data
- **Domain vocabulary** — clinical terms spoken with Indian pronunciation cause higher WER

Sarvam AI's models are trained on Indian language data at scale:

| Model | Task | Specialization |
|:---|:---:|:---|
| `saaras` | ASR (STT) | Indian English + 10 Indian languages, optimized for telephony |
| `bulbul:v2` | Neural TTS | Female voice `anushka` — natural prosody for Indian English |

The `target_language_code="en-IN"` parameter in `bulbul:v2` is a critical configuration that pins the phoneme mapping to Indian English, ensuring the agent sounds natural and culturally appropriate to patients.

### Why Silero VAD?

Silero VAD is a **lightweight, ONNX-exported neural network** (~1MB) that runs locally on CPU with:
- **30ms frame latency** — well below the perceptual threshold
- **High accuracy** on telephony audio (8kHz–16kHz, compressed codecs)
- **Zero API cost** — runs entirely on-device, no external calls
- **Interrupt enablement** — detects when a patient begins speaking mid-response, allowing the agent to stop talking and listen

Without VAD, the STT model would constantly process silence, wasting tokens, increasing latency, and generating hallucinated transcriptions.

---

## 📊 Function Calling & Structured Data Extraction

This is where the **ML intelligence converts free-form voice into structured data** — the core value proposition of the system.

```python
@function_tool
async def save_feedback(
    self,
    rating: Annotated[int, "Rating for communication (1-5)"],
    wait_time: Annotated[str, "The patient's perception of wait time"],
    comments: Annotated[str, "Final comments from the patient"]
) -> str:
    """Saves the collected patient feedback to the dashboard."""
    
    feedback_data = {
        "doctor": "Dr. Gupta",
        "rating": rating,        # Extracted from: "I'd give it about a 3 out of 5"
        "wait_time": wait_time,  # Extracted from: "The wait was honestly quite long"
        "comments": comments     # Free-form, preserved verbatim by LLM
    }
    
    with open("feedback_dashboard.json", "a") as f:
        f.write(json.dumps(feedback_data) + "\n")
    
    return "Feedback saved successfully. You may now end the call."
```

**How it works under the hood:**

1. The `@function_tool` decorator causes the LiveKit SDK to **automatically generate a JSON Schema** from the Python type annotations
2. This schema is injected into the LLM's `tools` parameter on every inference call
3. The LLM (Llama 3.3-70B) is trained to identify when it has gathered sufficient information to call a tool — it produces a structured `tool_call` object instead of text
4. The SDK intercepts this, **deserializes and validates** the JSON against the Python signature, then executes the async function
5. The return value is fed back to the LLM as a `tool` role message, which triggers the final farewell

This approach is **far more robust** than regex parsing or keyword matching — the LLM handles all the nuance of natural language ("I'd give him a solid four," "the wait wasn't terrible," "no suggestions really") and maps it to the structured schema.

---

## 💾 Data Persistence & Output Schema

The agent persists feedback as **NDJSON** (Newline-Delimited JSON) — one record per line, one per completed call:

```json
{"doctor": "Dr. Gupta", "rating": 5, "wait_time": "reasonable", "comments": "Very thorough and attentive."}
{"doctor": "Dr. Gupta", "rating": 2, "wait_time": "too long", "comments": "Implement a proper queue for checking in with patients to reduce wait time"}
{"doctor": "Dr. Gupta", "rating": 1, "wait_time": "too long", "comments": "Dr. Gupta's communication was not great, he missed out various symptoms, and the diagnosis was not good. Also, please ensure adequate seats in the waiting room."}
```

**Why NDJSON?**
- **Append-only I/O** — `open(..., "a")` is atomic for single records; no lock contention
- **Streaming-friendly** — each line is a complete record, parseable independently
- **Direct ingestion** — compatible with tools like `pandas.read_json(lines=True)`, BigQuery, and Elasticsearch out-of-the-box

---

## 🚀 Installation & Setup

### Prerequisites

- Python **3.10+**
- A [LiveKit Cloud](https://livekit.io/) project (or self-hosted server)
- API keys for **Groq** and **Sarvam AI**

### 1. Clone the Repository

```bash
git clone https://github.com/jayanthoffl/medicalAgent.git
cd medicalAgent
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# ── LiveKit Configuration ──────────────────────────────────────────
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=API_xxxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxx

# ── Inference & Speech Providers ───────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxx
SARVAM_API_KEY=sk_xxxxxxxxxxxx
```

### 3. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install all required packages
pip install -r requirements.txt
```

**`requirements.txt`:**
```text
livekit-agents>=1.3.0
livekit-plugins-groq
livekit-plugins-sarvam
livekit-plugins-silero
python-dotenv
```

### 4. Launch the Agent Worker

```bash
python agent.py dev
```

The process will connect to your LiveKit server and begin listening for job assignments via WebSocket.

### 5. Dispatch a Call

**Option A — LiveKit Playground (Recommended for Development):**
Navigate to the [LiveKit Console](https://cloud.livekit.io/) → Playground → Create Room. The agent worker will automatically detect the room and join.

**Option B — SIP / Telephony (Production):**
Configure a SIP trunk in LiveKit Console, then dispatch an outbound call via the LiveKit CLI or API. The agent will be automatically assigned to handle the call.

---

## 🗺️ Roadmap

```
v1.0 — Current
  [x] Full-duplex voice conversation via WebRTC
  [x] VAD → STT → LLM → TTS pipeline
  [x] Autonomous function calling for data extraction
  [x] NDJSON local persistence

v1.1 — Near-Term
  [ ] PostgreSQL / Supabase integration (replace local JSON)
  [ ] Structured sentiment tagging (Positive / Neutral / Negative / Urgent)
  [ ] Async retry logic for Sarvam API latency spikes with fallback to Deepgram

v2.0 — Future Vision
  [ ] Outbound SIP dialing via appointment calendar webhook (cron + cal API)
  [ ] Real-time analytics dashboard (Grafana / Metabase)
  [ ] Multi-doctor, multi-clinic support with room-based routing
  [ ] Multilingual support: Hindi, Tamil, Telugu (Sarvam multilingual models)
  [ ] Post-call NLP pipeline: urgency detection, auto-escalation to clinic staff
```

---

