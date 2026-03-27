## 🎹 piano2midi

Convert piano audio (MP3/WAV) into MIDI using AI (Spotify Basic Pitch).

---

### 🚀 Run

#### 1. First-time setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

#### 2. Quick run (if already set up)

```bash
source venv/bin/activate && python server.py
```


---

### 📂 Input

* MP3 / WAV audio file

### 📤 Output

* MIDI file
* Piano roll image
* Analysis report (JSON)

---

### ⚙️ Tech Stack

* basic-pitch (Spotify)
* pretty_midi
* numpy

---

### 📌 Notes

* Best results with clean piano recordings
* Processing time: ~1–3 minutes per track
