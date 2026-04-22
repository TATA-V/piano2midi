## 🎹 piano2midi

Convert piano audio (MP3/WAV) into MIDI using AI (Spotify Basic Pitch).

---

### 🚀 Run

#### 1. First-time setup

> **Note:** Python 3.11 is required. `basic-pitch` does not support Python 3.13 due to a numpy wheel incompatibility.
> Install it with: `brew install python@3.11`

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Start server

```bash
source venv/bin/activate && python server.py
```

Open **http://localhost:5050** in your browser.

---

### 📂 Input

* MP3 / WAV / OGG / FLAC / M4A (max 50MB)

### 📤 Output

* MIDI file
* Piano roll image
* Analysis report (key, BPM, dynamics, etc.)

---

### ⚙️ Tech Stack

* **basic-pitch** — AI transcription (Spotify)
* **Flask** — Web server
* **pretty_midi** — MIDI post-processing
* **matplotlib / scipy** — Visualization & analysis
* **librosa** — Audio processing

---

### 📌 Notes

* Best results with clean piano recordings
* Processing time: ~1–3 minutes per track
