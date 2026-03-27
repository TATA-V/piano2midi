"""
🎹 MP3 → MIDI Converter Server
Flask backend with basic-pitch AI transcription pipeline.
"""

import os
import uuid
import json
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Config ────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# In-memory job status store
jobs = {}


def update_job(job_id, **kwargs):
    """Update job status."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)


# ── Conversion Pipeline ──────────────────────────────────
def run_conversion(job_id, audio_path):
    """Run the full MP3 → MIDI conversion pipeline in a background thread."""
    try:
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)


        midi_raw_path = job_dir / "output_raw.mid"
        midi_clean_path = job_dir / "output_clean.mid"
        piano_roll_path = job_dir / "piano_roll.png"
        report_path = job_dir / "report.json"

        # ── Step 1: AI Transcription with basic-pitch ─────
        # basic-pitch accepts MP3/WAV/OGG/FLAC directly via librosa
        update_job(job_id, status="converting", step="🤖 AI 분석 중... (1~3분 소요)", progress=15)

        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        model_output, midi_data, note_events = predict(
            str(audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        # Save raw MIDI
        midi_data.write(str(midi_raw_path))
        update_job(job_id, step="AI 전사 완료", progress=60)

        # ── Step 3: MIDI Post-processing ──────────────────
        update_job(job_id, step="MIDI 후처리 중...", progress=70)

        import pretty_midi

        midi_raw = pretty_midi.PrettyMIDI(str(midi_raw_path))
        estimated_tempo = midi_raw.estimate_tempo()
        midi_clean = pretty_midi.PrettyMIDI(initial_tempo=estimated_tempo)

        removed, kept = 0, 0

        for inst in midi_raw.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name
            )
            notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
            vels = [n.velocity for n in notes]
            vel_thr = max(10, np.percentile(vels, 5)) if vels else 10
            seen = {}

            for n in notes:
                # Remove very short notes (ghost notes)
                if n.end - n.start < 0.04:
                    removed += 1
                    continue
                # Remove very quiet notes
                if n.velocity < vel_thr:
                    removed += 1
                    continue
                # Remove out-of-piano-range notes
                if n.pitch < 21 or n.pitch > 108:
                    removed += 1
                    continue
                # Remove duplicate notes too close together
                if n.pitch in seen and n.start - seen[n.pitch] < 0.05:
                    removed += 1
                    continue

                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(np.clip(n.velocity, 1, 127)),
                        pitch=n.pitch,
                        start=n.start,
                        end=min(n.end, n.start + 8.0),
                    )
                )
                seen[n.pitch] = n.start
                kept += 1

            new_inst.control_changes = inst.control_changes
            midi_clean.instruments.append(new_inst)

        midi_clean.write(str(midi_clean_path))
        update_job(job_id, step="MIDI 후처리 완료", progress=80)

        # ── Step 4: Piano Roll Visualization ──────────────
        update_job(job_id, step="피아노롤 생성 중...", progress=85)

        all_notes = sorted(
            [n for inst in midi_clean.instruments if not inst.is_drum for n in inst.notes],
            key=lambda n: n.start,
        )

        if all_notes:
            _generate_piano_roll(all_notes, str(piano_roll_path), jobs[job_id].get("title", ""))

        update_job(job_id, step="피아노롤 생성 완료", progress=90)

        # ── Step 5: Analysis Report ───────────────────────
        update_job(job_id, step="분석 리포트 생성 중...", progress=95)

        report = _generate_report(all_notes, midi_clean, estimated_tempo, removed, kept)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        update_job(
            job_id,
            status="done",
            step="변환 완료! 🎉",
            progress=100,
            report=report,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, status="error", error=str(e))


def _generate_piano_roll(notes, save_path, song_title=""):
    """Generate a beautiful dark-themed piano roll image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from collections import Counter
    from scipy.ndimage import uniform_filter1d

    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    starts = [n.start for n in notes]
    total_time = max(n.end for n in notes)

    BG, PANEL, GRID = "#0d0d1a", "#13132b", "#1e1e3a"
    TEXT, ACCENT = "#c8c8e8", "#ff6b9d"

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[3, 1.2, 1.2],
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    song_label = (song_title[:60] + "...") if len(song_title) > 60 else song_title
    fig.suptitle(f"🎹  {song_label}", color="white", fontsize=15, y=0.97)

    # ── Piano Roll ──
    ax_roll = fig.add_subplot(gs[0, :])
    ax_roll.set_facecolor(PANEL)
    ax_roll.set_title("Piano Roll", color=TEXT, fontsize=12, pad=8)

    pitch_min = max(min(pitches) - 3, 21)
    pitch_max = min(max(pitches) + 3, 108)
    black_offsets = {1, 3, 6, 8, 10}

    for p in range(pitch_min, pitch_max + 1):
        if p % 12 in black_offsets:
            ax_roll.axhspan(p - 0.5, p + 0.5, color="#09091a", alpha=0.5, zorder=0)

    note_names_all = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for oct_num in range(2, 9):
        c = (oct_num + 1) * 12
        if pitch_min <= c <= pitch_max:
            ax_roll.axhline(c, color=GRID, lw=0.7, zorder=1)
            ax_roll.text(
                -total_time * 0.005, c, f"C{oct_num}",
                color="#5555aa", fontsize=7.5, va="center", ha="right",
            )

    cmap = plt.cm.plasma
    for n in notes:
        rect = mpatches.FancyBboxPatch(
            (n.start, n.pitch - 0.42),
            max(n.end - n.start, 0.03), 0.84,
            boxstyle="round,pad=0.01",
            facecolor=cmap(n.velocity / 127),
            edgecolor="none", alpha=0.88, zorder=2,
        )
        ax_roll.add_patch(rect)

    ax_roll.set_xlim(0, total_time)
    ax_roll.set_ylim(pitch_min, pitch_max)
    ax_roll.set_xlabel("Time (sec)", color=TEXT, fontsize=9)
    ax_roll.set_ylabel("Pitch", color=TEXT, fontsize=9)
    ax_roll.tick_params(colors=TEXT, labelsize=8)
    for sp in ax_roll.spines.values():
        sp.set_edgecolor(GRID)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 127))
    cb = plt.colorbar(sm, ax=ax_roll, pad=0.01, shrink=0.85, aspect=30)
    cb.set_label("Velocity", color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    # ── Velocity Timeline ──
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_vel.set_facecolor(PANEL)
    ax_vel.set_title("Velocity (강약)", color=TEXT, fontsize=10, pad=6)
    ax_vel.scatter(starts, velocities, c=velocities, cmap="plasma", s=5, alpha=0.55, zorder=2)

    win = max(len(velocities) // 35, 3)
    if len(velocities) > win:
        sv = uniform_filter1d(np.array(velocities, dtype=float), size=win)
        ax_vel.plot(starts, sv, color=ACCENT, lw=1.6, label="Moving Avg", zorder=3)
        ax_vel.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, framealpha=0.7)

    ax_vel.set_xlim(0, total_time)
    ax_vel.set_ylim(0, 130)
    ax_vel.set_xlabel("Time (sec)", color=TEXT, fontsize=8)
    ax_vel.set_ylabel("Velocity", color=TEXT, fontsize=8)
    ax_vel.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_vel.spines.values():
        sp.set_edgecolor(GRID)

    # ── Polyphony Timeline ──
    ax_poly = fig.add_subplot(gs[1, 1])
    ax_poly.set_facecolor(PANEL)
    ax_poly.set_title("Polyphony (동시 발음 수)", color=TEXT, fontsize=10, pad=6)

    t_samples = np.linspace(0, total_time, 800)
    poly_vals = np.array([sum(1 for n in notes if n.start <= t < n.end) for t in t_samples])

    ax_poly.fill_between(t_samples, poly_vals, alpha=0.4, color="#7b5ea7", zorder=2)
    ax_poly.plot(t_samples, poly_vals, color="#b08fd8", lw=1.0, zorder=3)
    ax_poly.set_xlim(0, total_time)
    ax_poly.set_ylim(0, max(poly_vals) + 1)
    ax_poly.set_xlabel("Time (sec)", color=TEXT, fontsize=8)
    ax_poly.set_ylabel("Simultaneous Notes", color=TEXT, fontsize=8)
    ax_poly.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_poly.spines.values():
        sp.set_edgecolor(GRID)

    # ── Pitch Distribution ──
    ax_dist = fig.add_subplot(gs[2, 0])
    ax_dist.set_facecolor(PANEL)
    ax_dist.set_title("Pitch Distribution (조성 분석)", color=TEXT, fontsize=10, pad=6)

    pc_counts = Counter(p % 12 for p in pitches)
    bar_colors = plt.cm.plasma(np.linspace(0.15, 0.9, 12))
    bars = ax_dist.bar(
        range(12),
        [pc_counts.get(i, 0) for i in range(12)],
        color=bar_colors, edgecolor="#0a0a18", lw=0.5,
    )
    for idx, _ in pc_counts.most_common(3):
        bars[idx].set_edgecolor(ACCENT)
        bars[idx].set_linewidth(1.8)

    ax_dist.set_xticks(range(12))
    ax_dist.set_xticklabels(note_names_all, color=TEXT, fontsize=8)
    ax_dist.set_ylabel("Count", color=TEXT, fontsize=8)
    ax_dist.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_dist.spines.values():
        sp.set_edgecolor(GRID)

    # ── Range Histogram ──
    ax_hist = fig.add_subplot(gs[2, 1])
    ax_hist.set_facecolor(PANEL)
    ax_hist.set_title("Pitch Range (음역 분포)", color=TEXT, fontsize=10, pad=6)

    import pretty_midi as pm
    ax_hist.hist(pitches, bins=40, color="#5b8dd9", edgecolor="#0a0a18", lw=0.4, alpha=0.85)
    mean_pitch = int(np.mean(pitches))
    ax_hist.axvline(
        mean_pitch, color=ACCENT, lw=1.5, linestyle="--",
        label=f"Mean: {pm.note_number_to_name(mean_pitch)}",
    )
    ax_hist.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, framealpha=0.7)
    ax_hist.set_xlabel("MIDI Note Number", color=TEXT, fontsize=8)
    ax_hist.set_ylabel("Count", color=TEXT, fontsize=8)
    ax_hist.tick_params(colors=TEXT, labelsize=7)
    for sp in ax_hist.spines.values():
        sp.set_edgecolor(GRID)

    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _generate_report(notes, midi_obj, estimated_tempo, removed, kept):
    """Generate analysis report JSON."""
    import pretty_midi as pm
    from collections import Counter

    if not notes:
        return {"error": "No notes detected"}

    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    total_time = max(n.end for n in notes)

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pc_counts = Counter(p % 12 for p in pitches)

    # Key estimation
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    pc_vec = np.array([pc_counts.get(i, 0) for i in range(12)], dtype=float)
    pc_vec /= pc_vec.sum() + 1e-9

    best_score, best_key, best_mode = -np.inf, 0, "Major"
    for root in range(12):
        for profile, mode in [(major_profile, "Major"), (minor_profile, "Minor")]:
            score = np.corrcoef(pc_vec, np.roll(profile, root))[0, 1]
            if score > best_score:
                best_score, best_key, best_mode = score, root, mode

    # Dynamics analysis
    density = len(notes) / total_time
    if density < 2:
        feel = "느리고 서정적"
    elif density < 4:
        feel = "보통"
    elif density < 7:
        feel = "빠름"
    else:
        feel = "매우 빠름 (기교적)"

    h1 = float(np.mean([n.velocity for n in notes if n.start < total_time / 2]))
    h2 = float(np.mean([n.velocity for n in notes if n.start >= total_time / 2]))
    if h2 - h1 > 5:
        dynamics = "후반 클라이맥스 (crescendo)"
    elif h2 - h1 < -5:
        dynamics = "전반 강조 (decrescendo)"
    else:
        dynamics = "균등한 강약"

    return {
        "totalNotes": len(notes),
        "removedNotes": removed,
        "duration": round(total_time, 1),
        "durationMin": round(total_time / 60, 1),
        "pitchRange": {
            "low": pm.note_number_to_name(min(pitches)),
            "high": pm.note_number_to_name(max(pitches)),
        },
        "meanPitch": pm.note_number_to_name(int(np.mean(pitches))),
        "estimatedKey": f"{note_names[best_key]} {best_mode}",
        "keyConfidence": round(float(best_score), 2),
        "estimatedBPM": round(estimated_tempo, 1),
        "density": round(density, 1),
        "densityFeel": feel,
        "dynamics": dynamics,
        "avgVelocity": round(float(np.mean(velocities))),
        "maxVelocity": int(max(velocities)),
        "minVelocity": int(min(velocities)),
    }


# ── API Routes ────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload an audio file (MP3/WAV)."""
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("mp3", "wav", "ogg", "flac", "m4a"):
        return jsonify({"error": "지원하지 않는 형식입니다 (MP3, WAV, OGG, FLAC, M4A)"}), 400

    job_id = str(uuid.uuid4())[:8]
    filename = f"{job_id}.{ext}"
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    title = file.filename.rsplit(".", 1)[0]

    jobs[job_id] = {
        "id": job_id,
        "title": title,
        "filename": file.filename,
        "status": "uploaded",
        "step": "업로드 완료",
        "progress": 0,
        "error": None,
        "report": None,
        "filepath": str(filepath),
    }

    return jsonify({"id": job_id, "title": title, "filename": file.filename})


@app.route("/api/convert/<job_id>", methods=["POST"])
def start_conversion(job_id):
    """Start the MIDI conversion."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] not in ("uploaded", "error"):
        return jsonify({"error": "이미 변환 중이거나 완료되었습니다"}), 400

    job["status"] = "converting"
    job["progress"] = 5
    job["step"] = "변환 시작..."
    job["error"] = None

    thread = threading.Thread(
        target=run_conversion,
        args=(job_id, job["filepath"]),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """Get conversion status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    return jsonify({
        "id": job["id"],
        "title": job["title"],
        "status": job["status"],
        "step": job["step"],
        "progress": job["progress"],
        "error": job.get("error"),
        "report": job.get("report"),
    })


@app.route("/api/download/<job_id>", methods=["GET"])
def download_midi(job_id):
    """Download the converted MIDI file."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    midi_path = OUTPUT_DIR / job_id / "output_clean.mid"
    if not midi_path.exists():
        return jsonify({"error": "MIDI 파일이 아직 생성되지 않았습니다"}), 404

    title = jobs[job_id].get("title", "output")
    return send_file(
        midi_path,
        mimetype="audio/midi",
        as_attachment=True,
        download_name=f"{title}.mid",
    )


@app.route("/api/piano-roll/<job_id>", methods=["GET"])
def get_piano_roll(job_id):
    """Get the piano roll image."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    img_path = OUTPUT_DIR / job_id / "piano_roll.png"
    if not img_path.exists():
        return jsonify({"error": "Piano roll이 아직 생성되지 않았습니다"}), 404

    return send_file(img_path, mimetype="image/png")


@app.route("/api/audio/<job_id>", methods=["GET"])
def get_audio(job_id):
    """Serve the original uploaded audio for playback."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    filepath = jobs[job_id].get("filepath")
    if not filepath or not Path(filepath).exists():
        return jsonify({"error": "오디오 파일을 찾을 수 없습니다"}), 404

    return send_file(filepath)


@app.route("/")
def serve_index():
    """Serve the frontend index.html."""
    return send_file("index.html")


@app.route("/style.css")
def serve_css():
    """Serve the CSS file."""
    return send_file("style.css")


@app.route("/main.js")
def serve_js():
    """Serve the JS file."""
    return send_file("main.js")


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎹 MP3 → MIDI Converter Server")
    print("   http://localhost:5050")
    app.run(debug=True, host="0.0.0.0", port=5050)
