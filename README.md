# 🎥 Royal Recorder  
**An advanced screen + webcam recorder with built-in trimming and FFmpeg integration.**  
**Professional • Polished • Stylish • Cinematic**

<p align="center">
  <img src="logo.png" width="120" alt="Royal Recorder Logo"/><br>
  <i>“Record it like royalty.”</i>
</p>

---

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <br>
  <a href="https://github.com/FFmpeg/FFmpeg"><img src="https://img.shields.io/badge/FFmpeg-Integrated-lightgrey.svg" alt="FFmpeg"></a>
  <br>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Windows%2010%2B-blue.svg" alt="Platform"></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/Status-Stable-success.svg" alt="Status"></a>
</p>

---

## 🎬 Preview

<p align="center">
  <img src="preview.gif" width="700" alt="Royal Recorder Demo GIF"/><br>
  <i>Live preview — record, trim, and export in seconds.</i>
</p>

> 💡 *(Add your actual demo GIF named `preview.gif` in the root folder — or use a screen capture to show how the app runs.)*

---

## ✨ Overview

**Royal Recorder** is a premium screen and webcam recording app built in **Python**.  
It blends power and simplicity — record your screen, capture webcam overlays, and merge them with crystal-clear audio in real-time.  
Designed for creators, professionals, and developers who value both **performance** and **presentation**.

---

## 🎯 Key Features

- 🖥️ **Screen + Webcam Capture** — Record both seamlessly with PiP (Picture-in-Picture).  
- 🎙️ **Audio Recording** — Capture microphone or system audio with perfect sync.  
- ✂️ **Trim & Edit** — Built-in FFmpeg-powered trimming for precision cuts.  
- ⚡ **Auto Merge** — Smart post-processing for flawless exports.  
- 🧠 **Pause & Resume** — Full control during long sessions.  
- 🕶️ **Overlay & Timer Display** — Real-time duration and FPS overlay.  
- 🎨 **Cinematic UI** — Minimalist, dark-themed design for professionals.  
- 🧱 **Offline Ready** — No cloud dependency, everything runs locally.  

---

## 🧩 Tech Stack

| Component | Library |
|------------|----------|
| UI / Controls | Tkinter |
| Screen + Camera | OpenCV |
| Audio | PyAudio |
| Processing | FFmpeg |
| Encoding | ffmpeg-python |
| Build | PyInstaller |

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/RoyalRecorder.git
cd RoyalRecorder
```

---

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # on Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the application:
```bash
python recordit.py
```
- Select Screen Only or Screen + Camera mode
- Choose your microphone
- Click Start Recording
- Stop when ready — your video and audio are auto-merged

>  *( Optional: Trim your final recording directly within the app.)*


---

## 🧱 Build to EXE (Optional)

To build as a standalone Windows .exe:
```bash
pyinstaller recordit.spec
```

---

Your executable will appear in:
```bash
dist/
└── recordit.exe
```

---

## 🗂️ Folder Structure

```bashRoyalRecorder/
│
├── recordit.py
├── recordit.spec
├── royal_version.txt
├── logo.png
├── preview.gif
├── ffmpeg.exe
├── requirements.txt
├── .gitignore
└── venv/

```

---

## 🧑‍💻 Developer Notes
 
- Default FFmpeg path → ```bash BASE_DIR/ffmpeg.exe ``` 
- Default output folder → project root
- Camera URL & configs → handled in ```bash config.json``` (ignored in .gitignore)
- Supports multiple webcam fallback indexes
  
---


## 🕶️ Cinematic Design Philosophy
>  *( “It’s not just about recording — it’s about presentation.”)*

Minimal motion, clean UI, and fluid transitions — Royal Recorder is built for creators who care about aesthetic experience as much as functionality.

---

## 📜 License

Licensed under the MIT License.
Feel free to use, modify, or distribute with attribution.

---


<p align="center"> <b>Royal Recorder</b><br> <i>“Crafted with precision. Recorded with pride.”</i> 🎞️ </p> ```


