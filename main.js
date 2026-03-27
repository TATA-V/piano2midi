/**
 * 🎹 Piano2MIDI — Frontend Controller
 * Handles file upload, conversion progress, and result display.
 */

const API_BASE = "/api";

// ── State ─────────────────────────────────────────
let currentJobId = null;
let pollInterval = null;

// ── DOM Elements ──────────────────────────────────
const uploadSection = document.getElementById("upload-section");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const fileSection = document.getElementById("file-section");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const removeFileBtn = document.getElementById("remove-file-btn");
const convertBtn = document.getElementById("convert-btn");
const audioPlayer = document.getElementById("audio-player");
const progressSection = document.getElementById("progress-section");
const progressStep = document.getElementById("progress-step");
const progressBar = document.getElementById("progress-bar");
const progressPercent = document.getElementById("progress-percent");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const retryBtn = document.getElementById("retry-btn");
const resultSection = document.getElementById("result-section");
const downloadBtn = document.getElementById("download-btn");
const newConvertBtn = document.getElementById("new-convert-btn");
const reportGrid = document.getElementById("report-grid");
const pianoRollImg = document.getElementById("piano-roll-img");
const midiPlayer = document.getElementById("midi-player");

// ── Helpers ───────────────────────────────────────
function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function showSection(section) {
    [uploadSection, fileSection, progressSection, errorSection, resultSection].forEach(
        (s) => s.classList.add("hidden")
    );
    section.classList.remove("hidden");
}

// ── Upload ────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["mp3", "wav", "ogg", "flac", "m4a"].includes(ext)) {
        alert("지원하지 않는 형식입니다. MP3, WAV, OGG, FLAC, M4A 파일을 선택하세요.");
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        alert("파일 크기가 50MB를 초과합니다.");
        return;
    }

    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);

    // Set up audio player
    const audioUrl = URL.createObjectURL(file);
    audioPlayer.src = audioUrl;

    // Upload to server
    convertBtn.disabled = true;
    convertBtn.innerHTML = '<span class="btn-icon">⏳</span><span>업로드 중...</span>';

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || "업로드 실패");
        }

        const data = await res.json();
        currentJobId = data.id;

        convertBtn.disabled = false;
        convertBtn.innerHTML = '<span class="btn-icon">⚡</span><span>MIDI 변환 시작</span>';

        showSection(fileSection);
        uploadSection.classList.remove("hidden");
        uploadSection.classList.add("hidden");
        fileSection.classList.remove("hidden");
    } catch (err) {
        alert("업로드 실패: " + err.message);
        convertBtn.disabled = false;
        convertBtn.innerHTML = '<span class="btn-icon">⚡</span><span>MIDI 변환 시작</span>';
    }
}

// ── Remove File ───────────────────────────────────
removeFileBtn.addEventListener("click", () => {
    currentJobId = null;
    fileInput.value = "";
    audioPlayer.src = "";
    showSection(uploadSection);
});

// ── Convert ───────────────────────────────────────
convertBtn.addEventListener("click", async () => {
    if (!currentJobId) return;

    try {
        const res = await fetch(`${API_BASE}/convert/${currentJobId}`, { method: "POST" });
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || "변환 시작 실패");
        }

        fileSection.classList.add("hidden");
        progressSection.classList.remove("hidden");
        startPolling();
    } catch (err) {
        alert("변환 시작 실패: " + err.message);
    }
});

// ── Poll Status ───────────────────────────────────
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollStatus, 1000);
}

async function pollStatus() {
    if (!currentJobId) return;

    try {
        const res = await fetch(`${API_BASE}/status/${currentJobId}`);
        const data = await res.json();

        progressStep.textContent = data.step || "처리 중...";
        progressBar.style.width = data.progress + "%";
        progressPercent.textContent = data.progress + "%";

        if (data.status === "done") {
            clearInterval(pollInterval);
            pollInterval = null;
            showResult(data);
        } else if (data.status === "error") {
            clearInterval(pollInterval);
            pollInterval = null;
            showError(data.error);
        }
    } catch (err) {
        console.error("Poll error:", err);
    }
}

// ── Show Error ────────────────────────────────────
function showError(msg) {
    errorMessage.textContent = msg || "알 수 없는 오류가 발생했습니다";
    progressSection.classList.add("hidden");
    errorSection.classList.remove("hidden");
}

retryBtn.addEventListener("click", () => {
    errorSection.classList.add("hidden");
    fileSection.classList.remove("hidden");
});

// ── Show Result ───────────────────────────────────
function showResult(data) {
    progressSection.classList.add("hidden");
    resultSection.classList.remove("hidden");

    // Piano Roll Image
    pianoRollImg.src = `${API_BASE}/piano-roll/${currentJobId}`;

    // MIDI Player
    midiPlayer.src = `${API_BASE}/download/${currentJobId}`;

    // Analysis Report
    if (data.report) {
        renderReport(data.report);
    }
}

function renderReport(report) {
    const items = [
        { label: "총 음표 수", value: report.totalNotes?.toLocaleString(), unit: "개", color: "accent" },
        { label: "연주 시간", value: report.durationMin, unit: "분", color: "" },
        { label: "추정 BPM", value: report.estimatedBPM, unit: "", color: "pink" },
        { label: "추정 조성", value: report.estimatedKey, unit: `(${(report.keyConfidence * 100).toFixed(0)}%)`, color: "cyan" },
        { label: "음역대", value: `${report.pitchRange?.low} ~ ${report.pitchRange?.high}`, unit: "", color: "" },
        { label: "평균 음정", value: report.meanPitch, unit: "", color: "" },
        { label: "연주 밀도", value: report.density, unit: `음/초 · ${report.densityFeel}`, color: "green" },
        { label: "강약 흐름", value: report.dynamics, unit: "", color: "amber" },
        { label: "Velocity", value: report.avgVelocity, unit: `(${report.minVelocity}~${report.maxVelocity})`, color: "" },
        { label: "제거된 노이즈", value: report.removedNotes?.toLocaleString(), unit: "개", color: "" },
    ];

    reportGrid.innerHTML = items
        .map(
            (item) => `
        <div class="report-item">
            <div class="report-label">${item.label}</div>
            <div class="report-value ${item.color}">
                ${item.value}${item.unit ? `<small>${item.unit}</small>` : ""}
            </div>
        </div>
    `
        )
        .join("");
}

// ── Download ──────────────────────────────────────
downloadBtn.addEventListener("click", () => {
    if (!currentJobId) return;
    window.open(`${API_BASE}/download/${currentJobId}`, "_blank");
});

// ── New Convert ───────────────────────────────────
newConvertBtn.addEventListener("click", () => {
    currentJobId = null;
    fileInput.value = "";
    audioPlayer.src = "";
    reportGrid.innerHTML = "";
    pianoRollImg.src = "";
    midiPlayer.src = "";
    showSection(uploadSection);
});
