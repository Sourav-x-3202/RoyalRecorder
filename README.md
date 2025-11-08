#  Royal Recorder  
**An advanced screen + webcam recorder with built-in trimming and FFmpeg integration.**  
**Professional • Polished • Stylish • Cinematic**

<p align="center">
  <img width="243" height="227" alt="Royal Recorder Logo" src="https://github.com/user-attachments/assets/9d8712f7-835b-460a-9139-e68a66aa1c40" /><br>
  <i>“Record it like royalty.”</i>
</p>
<p align="center"><i>Where performance meets presentation.</i></p>

---

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://github.com/FFmpeg/FFmpeg"><img src="https://img.shields.io/badge/FFmpeg-Integrated-lightgrey.svg" alt="FFmpeg"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Windows%2010%2B-blue.svg" alt="Platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Stable-success.svg" alt="Status"></a>
</p>

---
##  Table of Contents
- [ See Royal Recorder in Action](#-see-royal-recorder-in-action)
- [ Screenshots](#️-screenshots)
- [ Overview](#-overview)
- [ How It Works](#️-how-it-works)
- [ Key Features](#-key-features)
- [ Tech Stack](#-tech-stack)
- [ Installation](#-installation)
- [ Usage](#️-usage)
- [ Build to EXE (Optional)](#-build-to-exe-optional)
- [ Folder Structure](#-folder-structure)
- [ Developer Notes](#-developer-notes)
- [ Roadmap](#️-roadmap)
- [ Contribute](#-contribute)
- [ Cinematic Design Philosophy](#-cinematic-design-philosophy)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Preview


<p align="center">
  <a href="https://github.com/user-attachments/assets/f7c6fc3d-a908-40d5-b001-5244fb7cbdb6" target="_blank">
    <img src="https://github.com/user-attachments/assets/7c63cf3e-190e-4e24-859e-c838e88f67c6" width="500" alt="🎬 See Royal Recorder in Action"/>
  </a><br>
  <p align="center">
  <a href="https://github.com/user-attachments/assets/f7c6fc3d-a908-40d5-b001-5244fb7cbdb6">
    <img src="https://img.shields.io/badge/▶️ Watch_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch Demo"/>
  </a><br>
  </p>
  <p align="center"><i>Click the image above to watch the live demo — record, trim, and export in seconds.</i></p>
</p>



---


##  Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/47a13d7a-f4af-4602-8131-08c9abe9d8a9" width="700" alt="Royal Recorder UI Preview"/>
</p>


---

##  Overview

**Royal Recorder** is a premium screen and webcam recording app built in **Python**.  
It blends power and simplicity — record your screen, capture webcam overlays, and merge them with crystal-clear audio in real-time.  
Designed for creators, professionals, and developers who value both **performance** and **presentation**.

---

##  How It Works  
<p align="center">
  <img src="https://github.com/user-attachments/assets/21fb7e31-4637-4732-b533-17eb2a6d943c" width="800" alt="How Royal Recorder Works"/>
</p>

1. **Capture** — Records screen + webcam simultaneously  
2. **Process** — Uses FFmpeg to merge and sync audio/video  
3. **Export** — Auto-saves your polished recording  

---


##  Key Features

-  **Screen + Webcam Capture** — Record both seamlessly with PiP (Picture-in-Picture).  
-  **Audio Recording** — Capture microphone or system audio with perfect sync.  
-  **Trim & Edit** — Built-in FFmpeg-powered trimming for precision cuts.  
-  **Auto Merge** — Smart post-processing for flawless exports.  
-  **Pause & Resume** — Full control during long sessions.  
-  **Overlay & Timer Display** — Real-time duration and FPS overlay.  
-  **Cinematic UI** — Minimalist, dark-themed design for professionals.  
-  **Offline Ready** — No cloud dependency, everything runs locally.  

---

##  Tech Stack

| Component | Library |
|------------|----------|
| UI / Controls | Tkinter |
| Screen + Camera | OpenCV |
| Audio | PyAudio |
| Processing | FFmpeg |
| Encoding | ffmpeg-python |
| Build | PyInstaller |

---

##  Installation

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
requirements.txt includes:
- opencv-python
- pyaudio
- ffmpeg-python
- pillow
- pyinstaller
- tkinter (preinstalled with Python)

---

###  FFmpeg Setup

This app requires [FFmpeg](https://ffmpeg.org/download.html).

1. Download FFmpeg and place the binary (`ffmpeg.exe` on Windows) in the same folder as this script, or add it to your system PATH.
2. The app will automatically detect and use it at runtime.


---

##  Usage
Run the application:
```bash
python recordit.py
```
- Select Screen Only or Screen + Camera mode
- Choose your microphone
- Click Start Recording
- Stop when ready — your video and audio are auto-merged

>  *( **Tip:** Trim your final recording directly inside the app — no need for extra tools.)*


---

##  Build to EXE (Optional)

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

##  Folder Structure

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

---

##  Quick Start

1. Launch the app:
   ```bash
   python recordit.py
   ```
2. Choose:

    - Screen only or Screen + Camera
    - Your Microphone

3. Hit Start Recording

4. Trim and export your final clip instantly.

> Your videos are auto-saved with timestamps for easy access!

---

## Add a “Config Example”
Below your Config Table:
### Example `config.json`
```json
{
  "ffmpeg_path": "C:/Tools/FFmpeg/bin/ffmpeg.exe",
  "output_path": "D:/Recordings/",
  "camera_index": 1,
  "resolution": "1920x1080",
  "frame_rate": 60
}
```
---

##  Developer Notes
 
- Default FFmpeg path → ```bash BASE_DIR/ffmpeg.exe ``` 
- Default output folder → project root
- Camera URL & configs → handled in ```bash config.json``` (ignored in .gitignore)
- Supports multiple webcam fallback indexes

<details>
  <summary> Developer Notes (Click to expand)</summary>

###  Core Configuration

- **Default FFmpeg Path** → `BASE_DIR/ffmpeg.exe`  
  The app automatically looks for the FFmpeg executable in the project’s root directory.  
  If it’s not found, it will attempt to use the system-wide FFmpeg installation (if added to PATH).  
  You can override this by specifying a custom path in the `config.json` file.  

  Example:
  ```json
  {
    "ffmpeg_path": "C:/Tools/FFmpeg/bin/ffmpeg.exe"
  }
  ```
 ### Output Handling

- **Default Output Directory** → Project root (`BASE_DIR`)
  All recordings (video, audio, and merged files) are saved automatically in the main project directory unless otherwise specified.
  The app also creates timestamped filenames to prevent accidental overwriting.

  Example output file:
  `Royal_Recording_2025-11-08_14-32-07.mp4`
  
- You can modify the output folder by adding this in `config.json`:
  ```json
  {
  "output_path": "D:/Recordings/"
  }
  ```
 ### Camera & Input Configuration
 - **Webcam Indexing:**
   The app dynamically scans connected cameras using OpenCV (`cv2.VideoCapture`) and automatically picks the first available webcam.
   If your system has multiple cameras, you can specify which one to use in `config.json`:
  ```json
  {
  "camera_index": 1
  }

  ```
 - **Fallback Logic:**
   If the specified camera index fails, Royal Recorder falls back to index `0` automatically to ensure recording continuity.
   
 - **Microphone Selection:**
   Audio input is handled through PyAudio. The app lists available devices in the dropdown menu — you can select your preferred mic at
   runtime.

### 🧩 Config File Summary (`config.json`)

| Key | Description | Example |
|------|-------------|----------|
| `ffmpeg_path` | Custom FFmpeg binary location | `"C:/Tools/FFmpeg/bin/ffmpeg.exe"` |
| `output_path` | Folder where all recordings are saved | `"D:/Recordings/"` |
| `camera_index` | Webcam index (0 = default) | `1` |
| `resolution` | Output resolution (optional) | `"1920x1080"` |
| `frame_rate` | Target FPS (optional, default = 30) | `60` |


 ### Runtime Behavior

  - The app checks the validity of paths and devices at startup.
  - Missing or invalid paths are handled gracefully — with a notification to the user.
  - Temporary files (for trimming/merging) are auto-cleaned after export.
  - All video/audio merging is powered by FFmpeg subprocess calls wrapped in Python (`ffmpeg-python`).
    

 ### Extensibility

  - The `config.json` structure is modular — you can easily extend it for:
  - Custom overlays
  - Hotkey bindings
  - Auto-upload destinations (future roadmap)
   
</details> ```


  
---

## Roadmap

- Auto-update system
- AI-powered smart editing (auto highlight removal)
- Cloud backup integration
- Mac/Linux support
- UI dark/light theme toggle

   
---

## Contribute

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or report a bug.

If you like this project, don’t forget to star the repo to support development!
  
---

##  Cinematic Design Philosophy
>  *( “It’s not just about recording — it’s about presentation.”)*

Minimal motion, clean UI, and fluid transitions — Royal Recorder is built for creators who care about aesthetic experience as much as functionality.

---

##  Acknowledgments
- Inspired by open-source screen recorders and UI frameworks.
- Built  by Sourav Sharma.

---

##  License

Licensed under the MIT License.
Feel free to use, modify, or distribute with attribution.

---


<p align="center"> <b>Royal Recorder</b><br> <i>“Crafted with precision. Recorded with pride.”</i> </p> 
