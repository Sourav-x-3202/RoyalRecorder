"""
doit.py - Royal Recorder Pro (final polished)
- Single-file screen+camera recorder with:
  * Improved audio/video sync using timestamped audio blocks (Option A)
  * Pause/Resume, webcam switching, FPS display
  * Separate-audio saving option and merged output option
  * Background merging, trimming, hotkeys, overlay improvements
  * Microphone selection dialog
  * Timestamped output filenames & choose folder via Browse
  * Keeps UI flow and concept intact, polished layout & bug fixes
"""
import ctypes
import os
import sys
import time
import threading
import subprocess
import queue
from pathlib import Path
import cv2
import numpy as np
import pyautogui
import sounddevice as sd
from scipy.io.wavfile import write
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
import json, os
from pathlib import Path
from tkinter import messagebox, filedialog, simpledialog

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"recordit")

# ================ CONFIG =================

import json, os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# Load camera config safely
CONFIG_PATH = BASE_DIR / "config.json"
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        DEFAULT_CAM_URL = config.get("DEFAULT_CAM_URL", 0)
    except Exception:
        DEFAULT_CAM_URL = 0
else:
    DEFAULT_CAM_URL = 0  # fallback if config missing

CAM_FALLBACK_INDEXES = [0, 1, 2]

# Default output folder (user can change via UI)
OUTPUT_FOLDER = BASE_DIR

# Temp file names (per-run these will be updated)
TMP_VIDEO = OUTPUT_FOLDER / "tmp_video.avi"
TMP_AUDIO = OUTPUT_FOLDER / "tmp_audio.wav"
OUTPUT_VIDEO = OUTPUT_FOLDER / "royal_recording.mp4"

FFMPEG_LOCAL_PATH = BASE_DIR / "ffmpeg.exe"
FPS = 20
MIC_RATE = 44100
CAM_WIDTH = 1280
CAM_HEIGHT = 720

LOGO_FILENAME = BASE_DIR / "logo.png"


# ================ GLOBALS =================
_camera_frame_lock = threading.Lock()
_camera_frame = None  # BGR frame

_recording_flag = False
_paused_flag = False

# audio_queue will carry tuples (timestamp_seconds, numpy_array)
audio_queue = queue.Queue()
_audio_capture_running = False

_selected_input_device = None

_preview_lock = threading.Lock()

# small counters for FPS
_frames_written_lock = threading.Lock()
_frames_written = 0

# store first timestamps discovered for alignment
_last_video_start_ts = None  # set when recording_loop starts
_first_audio_ts = None

# audio/video offset (audio_first_ts - video_start_ts), positive => audio started later
_audio_video_offset = 0.0

# ================ UTIL FUNCTIONS =================
def debug_print(*args, **kwargs):
    print("[RoyalRecorder]", *args, **kwargs)

def ensure_dirs():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def shutil_which(name):
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None

def ffmpeg_path():
    if FFMPEG_LOCAL_PATH.exists():
        return str(FFMPEG_LOCAL_PATH)
    if shutil_which("ffmpeg"):
        return "ffmpeg"
    return None

def _timestamped_output_name(base: Path, prefix="royal_recording"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base / f"{prefix}_{ts}.mp4"

# ================ AUDIO DEVICE HELPERS =================
def query_audio_devices():
    """Return list of (index, name) for input-capable devices."""
    try:
        devs = sd.query_devices()
        res = []
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                res.append((i, d["name"]))
        return res
    except Exception as e:
        debug_print("query_audio_devices error:", e)
        return []

def select_input_device(prefer_headset=True):
    """
    Auto-select a device index. Returns None if default is desired.
    """
    debug_print("Selecting audio input device (prefer_headset=%s)" % prefer_headset)
    try:
        devices = sd.query_devices()
        default_input = None
        try:
            di = sd.default.device
            if isinstance(di, (list, tuple)) and di:
                default_input = di[0]
            elif isinstance(di, int):
                default_input = di
        except Exception:
            default_input = None
    except Exception as e:
        debug_print("sounddevice query_devices error:", e)
        devices = []
        default_input = None

    headset_keywords = ("headphone", "headset", "usb", "mic", "microphone", "airpods", "blue")
    if prefer_headset and devices:
        for i, d in enumerate(devices):
            try:
                name = d['name'].lower()
                max_input = d.get('max_input_channels', 0)
                if max_input > 0 and any(k in name for k in headset_keywords):
                    debug_print("Found headset-like device:", i, d['name'])
                    return i
            except Exception:
                continue

    if default_input is not None:
        debug_print("Using default input device:", default_input)
        return default_input

    for i, d in enumerate(devices):
        try:
            if d.get('max_input_channels', 0) > 0:
                debug_print("Using first available input device:", i, d['name'])
                return i
        except Exception:
            continue

    debug_print("No suitable input device found.")
    return None

# ================ CAPTURE LOOPS =================
def try_open_camera_stream(cam_url):
    debug_print("Attempting to open camera stream:", cam_url)
    if cam_url:
        try:
            cap = cv2.VideoCapture(cam_url)
            time.sleep(0.3)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                debug_print("Opened camera (HTTP stream).")
                return cap, "Wi-Fi stream"
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            debug_print("Exception opening HTTP camera:", e)

    for idx in CAM_FALLBACK_INDEXES:
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            time.sleep(0.25)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                debug_print(f"Opened camera (local index {idx}).")
                return cap, f"Local cam index {idx}"
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            debug_print("Exception trying camera index", idx, e)
    debug_print("No camera found for given URL/indexes.")
    return None, None

def camera_capture_loop(cam_url, stop_event, preview_callback=None, overlay_logo=None, cam_index=None):
    """
    Continuously capture frames. Use cam_index if provided to force local index.
    """
    global _camera_frame
    # If cam_url is a local index sentinel, attempt that first
    if cam_index is not None:
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(cam_index))
            time.sleep(0.25)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap, desc = try_open_camera_stream(cam_url)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                desc = f"Local cam index {cam_index}"
        except Exception:
            cap, desc = try_open_camera_stream(cam_url)
    else:
        cap, desc = try_open_camera_stream(cam_url)

    if not cap:
        debug_print("camera_capture_loop: no camera available, exiting thread.")
        return

    debug_print("camera_capture_loop started. Camera description:", desc)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.06)
                continue
            with _camera_frame_lock:
                _camera_frame = frame.copy()
            if preview_callback:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if overlay_logo:
                        try:
                            logo = overlay_logo.copy().convert("RGBA")
                            alpha = logo.split()[-1].point(lambda p: int(p * 0.7))
                            logo.putalpha(alpha)
                            pil.paste(logo, (8, 8), logo)
                        except Exception as _e:
                            debug_print("Failed to paste overlay logo in camera preview:", _e)
                    preview_callback(pil)
                except Exception as e:
                    debug_print("camera_capture_loop preview_callback error:", e)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    debug_print("camera_capture_loop finished.")

def audio_capture_loop(stop_event, device=None, enable_audio=True):
    """
    Record audio chunks and push (timestamp, chunk) into audio_queue.
    """
    global audio_queue, _audio_capture_running
    debug_print("audio_capture_loop starting (device=%s enable_audio=%s)..." % (str(device), enable_audio))
    _audio_capture_running = True

    def callback(indata, frames, time_info, status):
        # time_info timestamp isn't always reliable across platforms; use time.time()
        ts = time.time()
        if status:
            debug_print("Sounddevice status:", status)
        if stop_event.is_set():
            return
        if not enable_audio:
            return
        try:
            # we copy to make a contiguous array (float32)
            audio_queue.put((ts, indata.copy()))
        except Exception as e:
            debug_print("audio_capture_loop queue put error:", e)

    try:
        # open with channels=1 (mono) if possible
        channels = 1
        with sd.InputStream(samplerate=MIC_RATE, channels=channels, dtype='float32', callback=callback, device=device):
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        debug_print("audio_capture_loop exception:", e)
    _audio_capture_running = False
    debug_print("audio_capture_loop ended.")

# ================ RECORDING LOOP =================
def recording_loop(mode, stop_event, overlay_state_getter, fps_cfg=FPS, enable_audio=True):
    """
    Recording loop writes the video file, collects audio (timestamped) from audio_queue,
    and writes aligned TMP_AUDIO at the end using timestamps for alignment.
    """
    global _camera_frame, _frames_written, _last_video_start_ts, _first_audio_ts, _audio_video_offset
    debug_print("recording_loop started with mode:", mode)
    screen_w, screen_h = pyautogui.size()
    debug_print("Screen size:", screen_w, screen_h)

    # ensure tmp files removed
    try:
        if TMP_VIDEO.exists():
            TMP_VIDEO.unlink()
        if TMP_AUDIO.exists():
            TMP_AUDIO.unlink()
    except Exception:
        pass

    # try video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(str(TMP_VIDEO), fourcc, fps_cfg, (screen_w, screen_h))
    if not vw.isOpened():
        debug_print("VideoWriter XVID failed, trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(str(TMP_VIDEO), fourcc, fps_cfg, (screen_w, screen_h))
    if not vw.isOpened():
        debug_print("VideoWriter failed to open. Aborting recording.")
        return
    debug_print("VideoWriter opened:", TMP_VIDEO)

    audio_blocks = []  # list of (ts, ndarray)
    frame_interval = 1.0 / float(fps_cfg) if fps_cfg > 0 else 1.0 / FPS
    last_time = time.time()
    frames_written_local = 0
    _last_video_start_ts = time.time()
    _first_audio_ts = None

    try:
        while not stop_event.is_set():
            # pause support: if paused, don't write frames but keep sleeping to maintain time base
            if _paused_flag:
                time.sleep(0.05)
                last_time = time.time()
                continue

            target_time = last_time + frame_interval
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = target_time

            if mode == "Screen + Camera":
                screenshot = pyautogui.screenshot()
                base = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            else:
                with _camera_frame_lock:
                    camf = _camera_frame.copy() if _camera_frame is not None else None
                if camf is None:
                    base = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                else:
                    base = cv2.resize(camf, (screen_w, screen_h))

            # PIP overlay if applicable
            with _camera_frame_lock:
                camf = _camera_frame.copy() if _camera_frame is not None else None
            if mode == "Screen + Camera" and camf is not None:
                try:
                    state = overlay_state_getter()
                    ox, oy, ow, oh = int(state["x"]), int(state["y"]), int(state["w"]), int(state["h"])
                    ox = max(0, min(screen_w - 1, ox))
                    oy = max(0, min(screen_h - 1, oy))
                    ow = max(1, min(screen_w - ox, ow))
                    oh = max(1, min(screen_h - oy, oh))
                    small = cv2.resize(camf, (ow, oh))
                    base[oy:oy+oh, ox:ox+ow] = small
                except Exception as e:
                    debug_print("Overlay paste error:", e)

            vw.write(base)
            frames_written_local += 1
            with _frames_written_lock:
                _frames_written = frames_written_local

            # collect audio blocks (non-blocking)
            while not audio_queue.empty():
                try:
                    item = audio_queue.get_nowait()
                    if isinstance(item, tuple) and len(item) == 2:
                        ts_block, blk = item
                        if _first_audio_ts is None:
                            _first_audio_ts = ts_block
                        audio_blocks.append((ts_block, blk))
                    else:
                        # fallback, treat item as raw array (old style) and timestamp it now
                        audio_blocks.append((time.time(), item))
                except Exception:
                    break

    except Exception as e:
        debug_print("Exception inside recording_loop:", e)
    finally:
        try:
            vw.release()
            debug_print("Video writer released. Frames written:", frames_written_local)
        except Exception as e:
            debug_print("Error releasing video writer:", e)

        # Now align audio using timestamps
        try:
            # convert audio_blocks to single mono array aligned to video start
            if audio_blocks and enable_audio:
                # Combine blocks in chronological order
                audio_blocks.sort(key=lambda x: x[0])
                # Flatten into 1D float32 array and record their timestamps
                chunks = []
                times = []
                for ts_block, blk in audio_blocks:
                    # blk may be shape (N, channels)
                    arr = blk
                    if arr.ndim > 1 and arr.shape[1] > 1:
                        arr_mono = arr.mean(axis=1)
                    else:
                        arr_mono = arr.flatten()
                    chunks.append(arr_mono)
                    times.append(ts_block)

                # Concatenate
                audio_np = np.concatenate(chunks, axis=0)
                audio_start_ts = times[0]
                video_start_ts = _last_video_start_ts or time.time()
                # compute offset (audio_first - video_start)
                offset = audio_start_ts - video_start_ts
                debug_print(f"Audio first ts={audio_start_ts}, video start={video_start_ts}, offset={offset:.4f}s")

                # store global for merge worker
                try:
                    global _audio_video_offset
                    _audio_video_offset = float(offset)
                except Exception:
                    _audio_video_offset = 0.0

                # Convert float32 [-1,1] to int16
                audio_int16 = np.int16(np.clip(audio_np * 32767, -32768, 32767))

                # Determine expected length using frames_written_local and fps_cfg
                video_duration_sec = frames_written_local / float(fps_cfg) if fps_cfg > 0 else len(audio_int16) / float(MIC_RATE)
                expected_samples = int(round(video_duration_sec * MIC_RATE))

                # If audio started after video (offset > 0), pad front
                if offset > 0.005:
                    pad_samples = int(round(offset * MIC_RATE))
                    debug_print(f"Padding front with {pad_samples} samples because audio started after video")
                    audio_int16 = np.concatenate([np.zeros(pad_samples, dtype=np.int16), audio_int16])
                elif offset < -0.005:
                    # audio started earlier; trim front
                    trim_samples = int(round(-offset * MIC_RATE))
                    if trim_samples < len(audio_int16):
                        debug_print(f"Trimming front by {trim_samples} samples because audio started before video")
                        audio_int16 = audio_int16[trim_samples:]
                    else:
                        # nothing left: make silence
                        audio_int16 = np.zeros((1,), dtype=np.int16)

                # Finally trim/pad to expected_samples
                if len(audio_int16) > expected_samples:
                    debug_print(f"Trimming audio from {len(audio_int16)} to expected {expected_samples} samples")
                    audio_int16 = audio_int16[:expected_samples]
                elif len(audio_int16) < expected_samples:
                    pad_len = expected_samples - len(audio_int16)
                    debug_print(f"Padding audio from {len(audio_int16)} to expected {expected_samples} samples (pad {pad_len})")
                    audio_int16 = np.concatenate([audio_int16, np.zeros(pad_len, dtype=np.int16)])

                write(str(TMP_AUDIO), MIC_RATE, audio_int16)
                debug_print("Audio written to (aligned):", TMP_AUDIO, "samples:", len(audio_int16))
            else:
                # no audio captured or audio disabled -> write small silent file whose length equals minimal expected
                expected_samples = max(1, int(round((frames_written_local / float(fps_cfg) if fps_cfg > 0 else 0.1) * MIC_RATE)))
                silent = np.zeros((expected_samples,), dtype=np.int16)
                write(str(TMP_AUDIO), MIC_RATE, silent)
                debug_print("No audio blocks captured; wrote short silent WAV:", TMP_AUDIO)
        except Exception as e:
            debug_print("Error writing audio file:", e)

    debug_print("recording_loop finished.")

# ================ UI / OVERLAY ================
class OverlayWindow(tk.Toplevel):
    def __init__(self, master, width=320, height=200, logo_image=None):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        try:
            self.wm_attributes("-alpha", 0.96)
        except Exception:
            pass
        self.config(bg="black")
        sw, sh = pyautogui.size()
        initial_x = sw - width - 20
        initial_y = 40
        self.geometry(f"{width}x{height}+{initial_x}+{initial_y}")
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self._img_tk = None
        self.logo_image = logo_image
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.timer_var = tk.StringVar(value="00:00:00")
        # Small labels inside overlay for FPS/duration
        self.canvas.create_text(10, 10, anchor="nw", fill="white", font=("Arial", 10), text="Preview", tag="overlay_label")
        self.fps_text_id = self.canvas.create_text(10, 26, anchor="nw", fill="white", font=("Arial", 9), text=self.fps_var.get(), tag="overlay_fps")
        self.time_text_id = self.canvas.create_text(10, 42, anchor="nw", fill="white", font=("Arial", 9), text=self.timer_var.get(), tag="overlay_time")

        # Dragging
        self._drag_data = {"x": 0, "y": 0}
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._center_top_right)

    def _on_press(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        new_x = self.winfo_x() + dx
        new_y = self.winfo_y() + dy
        self.geometry(f"+{new_x}+{new_y}")

    def _center_top_right(self, event=None):
        sw, sh = pyautogui.size()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"{w}x{h}+{sw - w - 20}+40")

    def update_preview(self, pil_img):
        try:
            w, h = self.winfo_width(), self.winfo_height()
            if w <= 0 or h <= 0:
                return
            img = pil_img.resize((w, h), Image.BICUBIC)
            if self.logo_image:
                try:
                    logo = self.logo_image.copy().convert("RGBA")
                    logo_w = max(28, int(w * 0.18))
                    aspect = logo.size[1] / logo.size[0] if logo.size[0] else 1
                    logo_h = int(logo_w * aspect)
                    logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
                    alpha = logo.split()[-1].point(lambda p: int(p * 0.8))
                    logo.putalpha(alpha)
                    img.paste(logo, (8, 8), logo)
                except Exception as e:
                    debug_print("Failed to paste overlay logo:", e)
            self._img_tk = ImageTk.PhotoImage(img)
            # draw background image at (0,0)
            self.canvas.create_image(0, 0, anchor="nw", image=self._img_tk)
            # update FPS and time labels (text items)
            try:
                self.canvas.itemconfig(self.fps_text_id, text=self.fps_var.get())
                self.canvas.itemconfig(self.time_text_id, text=self.timer_var.get())
            except Exception:
                pass
        except Exception as e:
            debug_print("Overlay update_preview error:", e)

    def set_fps(self, fps_value):
        try:
            self.fps_var.set(f"FPS: {fps_value:.2f}")
            self.canvas.itemconfig(self.fps_text_id, text=self.fps_var.get())
        except Exception:
            pass

    def set_time(self, hhmmss):
        try:
            self.timer_var.set(hhmmss)
            self.canvas.itemconfig(self.time_text_id, text=self.timer_var.get())
        except Exception:
            pass

    def get_state(self):
        return {"x": self.winfo_x(), "y": self.winfo_y(), "w": self.winfo_width(), "h": self.winfo_height()}

# ================ MAIN APP =================
class RoyalRecorderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Royal Recorder Pro")
        self.geometry("760x600")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # thread refs/events
        self._cam_thread = None
        self._cam_stop_event = None
        self._record_thread = None
        self._record_stop_event = None
        self._audio_thread = None
        self._audio_stop_event = None
        self._merge_thread = None

        # runtime state
        self.enable_audio = True
        self.enable_audio_var = tk.BooleanVar(value=self.enable_audio)
        self.save_separate_audio_var = tk.BooleanVar(value=False)
        self.elapsed_seconds = 0
        self._timer_job = None

        # selected mic & webcam
        self.selected_mic_index = None
        self.selected_cam_index = None  # None means use cam_url or fallback
        self.available_mics = query_audio_devices()

        # load logo as CTkImage to avoid high-DPI warning
        self.logo_pil = None
        self.logo_ctkimage = None
        if LOGO_FILENAME.exists():
            try:
                img = Image.open(LOGO_FILENAME).convert("RGBA")
                self.logo_pil = img
                try:
                    # customtkinter has CTkImage - create from PIL.Image
                    self.logo_ctkimage = ctk.CTkImage(light_image=img, size=(72, 72))
                except Exception:
                    # fallback: keep PIL->ImageTk but CTk warns on HiDPI
                    self.logo_ctkimage = None
            except Exception as e:
                debug_print("Failed to load logo.png:", e)

        # TOP layout
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=12, pady=(8,6))

        if self.logo_ctkimage:
            try:
                logo_lbl = ctk.CTkLabel(top_frame, image=self.logo_ctkimage, text="")
                logo_lbl.pack(side="left", padx=(6,12))
            except Exception as e:
                debug_print("Error creating CTk top logo widget:", e)
                # fallback to PIL
                if self.logo_pil:
                    logo_display = ImageTk.PhotoImage(self.logo_pil.resize((72,72), Image.Resampling.LANCZOS))
                    logo_lbl = ctk.CTkLabel(top_frame, image=logo_display, text="")
                    logo_lbl.image = logo_display
                    logo_lbl.pack(side="left", padx=(6,12))
        else:
            if self.logo_pil:
                try:
                    logo_display = ImageTk.PhotoImage(self.logo_pil.resize((72,72), Image.Resampling.LANCZOS))
                    logo_lbl = ctk.CTkLabel(top_frame, image=logo_display, text="")
                    logo_lbl.image = logo_display
                    logo_lbl.pack(side="left", padx=(6,12))
                except Exception as e:
                    debug_print("Error creating PIL top logo widget:", e)

        ctk.CTkLabel(top_frame, text="Royal Recorder Pro", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", anchor="n")
        ctk.CTkLabel(self, text="Screen + Camera recorder — Premium UI", font=ctk.CTkFont(size=11)).pack()

        # Camera URL + webcam select
        frm_cam = ctk.CTkFrame(self)
        frm_cam.pack(fill="x", padx=18, pady=(8, 6))
        ctk.CTkLabel(frm_cam, text="Camera Stream (Wi-Fi URL) or leave blank for USB virtual cam:", anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.cam_url_var = ctk.StringVar(value=DEFAULT_CAM_URL)
        self.cam_entry = ctk.CTkEntry(frm_cam, textvariable=self.cam_url_var, width=520)
        self.cam_entry.grid(row=1, column=0, padx=6, pady=(0,8), sticky="w")

        # webcam selection dropdown
        self.cam_options = ["Auto (URL or first available)"] + [f"Local index {i}" for i in CAM_FALLBACK_INDEXES]
        self.cam_select_var = ctk.StringVar(value=self.cam_options[0])
        self.cam_select_menu = ctk.CTkOptionMenu(frm_cam, values=self.cam_options, variable=self.cam_select_var, width=240)
        self.cam_select_menu.grid(row=1, column=1, padx=6, pady=(0,8), sticky="e")

        # Mode
        self.mode_var = ctk.StringVar(value="Screen + Camera")
        mode_menu = ctk.CTkOptionMenu(self, values=["Screen + Camera", "Camera Only"], variable=self.mode_var, width=220)
        mode_menu.pack(pady=(6,4))

        # overlay checkbox
        self.show_overlay_var = tk.BooleanVar(value=True)
        self.overlay_checkbox = ctk.CTkCheckBox(self, text="Always show camera preview (overlay)", variable=self.show_overlay_var)
        self.overlay_checkbox.pack(pady=(0,8))

        # Controls (aligned)
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=(6, 10), fill="x", padx=18)
        self.btn_preview = ctk.CTkButton(btn_frame, text="▶ Start Preview", width=180, command=self.start_camera_preview)
        self.btn_preview.grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.btn_switch_cam = ctk.CTkButton(btn_frame, text="Switch Webcam", width=140, command=self._switch_webcam)
        self.btn_switch_cam.grid(row=0, column=1, padx=8, pady=6)
        self.btn_pause = ctk.CTkButton(btn_frame, text="⏸ Pause", width=140, command=self._toggle_pause, state="disabled")
        self.btn_pause.grid(row=0, column=2, padx=8, pady=6)
        self.btn_start = ctk.CTkButton(btn_frame, text="🔴 Start Recording (F9)", width=200, fg_color="#FF3B3B", hover_color="#CC2E2E", command=self.start_recording)
        self.btn_start.grid(row=0, column=3, padx=8, pady=6, sticky="e")
        self.btn_stop = ctk.CTkButton(self, text="⏹ Stop Recording (F10)", width=720, command=self.stop_recording)
        self.btn_stop.pack(pady=(6, 8))

        # info row (status + timer)
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=18, pady=(6,10))
        self.status_var = ctk.StringVar(value="Ready")
        self.timer_var = tk.StringVar(value="Duration: 00:00:00 | FPS: 0.0")
        status_label = ctk.CTkLabel(info_frame, textvariable=self.status_var, font=ctk.CTkFont(size=12))
        status_label.pack(side="left")
        timer_label = ctk.CTkLabel(info_frame, textvariable=self.timer_var, font=ctk.CTkFont(size=12))
        timer_label.pack(side="right")

        # save path + trim
        path_frame = ctk.CTkFrame(self)
        path_frame.pack(fill="x", padx=18, pady=(6,10))
        self.out_path_label = ctk.CTkLabel(path_frame, text=f"Save to: {OUTPUT_FOLDER}", anchor="w")
        self.out_path_label.pack(side="left", padx=(6,6))
        ctk.CTkButton(path_frame, text="Choose Folder", width=160, command=self.choose_folder).pack(side="right", padx=(6,6))

        trim_frame = ctk.CTkFrame(self)
        trim_frame.pack(fill="x", padx=18, pady=(6,10))
        ctk.CTkButton(trim_frame, text="Trim Recording", width=200, command=self.trim_video_ui).pack(side="left", padx=6)
        ctk.CTkButton(trim_frame, text="Open Output Folder", width=200, command=lambda: os.startfile(str(OUTPUT_FOLDER))).pack(side="left", padx=6)

        # overlay window
        self.overlay_win = OverlayWindow(self, width=360, height=220, logo_image=self.logo_pil if self.logo_pil else None)
        if not self.show_overlay_var.get():
            self.overlay_win.withdraw()
        else:
            # keep overlay visible even when minimized as user requested (unless they uncheck)
            self.overlay_win.deiconify()

        # Settings menu: audio toggle, save separate, mic selection
        menubar = tk.Menu(self)
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_checkbutton(label="Enable Audio Recording", onvalue=True, offvalue=False, variable=self.enable_audio_var, command=self._on_audio_toggle)
        settings_menu.add_checkbutton(label="Save Separate Audio File", onvalue=True, offvalue=False, variable=self.save_separate_audio_var)
        settings_menu.add_separator()
        settings_menu.add_command(label="Select Microphone...", command=self._choose_mic_dialog)
        settings_menu.add_separator()
        settings_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Royal Recorder Pro — Final"))
        menubar.add_cascade(label="Settings", menu=settings_menu)
        try:
            self.config(menu=menubar)
        except Exception:
            pass

        # keyboard hotkeys
        self.bind_all("<F9>", lambda e: self.start_recording())
        self.bind_all("<F10>", lambda e: self.stop_recording())

        # minimize behavior
        self.bind("<Unmap>", lambda e: self._on_minimize(e))
        self.bind("<Map>", lambda e: self._on_restore(e))

        # graceful close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # record time helper
        self._record_start_time = None

    # ============== UI HANDLERS ==============
    def _on_audio_toggle(self):
        self.enable_audio = bool(self.enable_audio_var.get())
        debug_print("Audio toggled:", self.enable_audio)

    def _choose_mic_dialog(self):
        devs = query_audio_devices()
        if not devs:
            messagebox.showwarning("No microphones", "No input devices found.")
            return
        # Build options
        opts = [f"{i}: {n}" for (i, n) in devs]
        dlg = simpledialog.askstring("Choose mic", "Enter device index from list:\n" + "\n".join(opts))
        if dlg is None:
            return
        try:
            idx = int(dlg.strip())
            # verify
            names = dict(devs)
            if idx not in names:
                messagebox.showerror("Invalid", "Device index not found in list.")
                return
            self.selected_mic_index = idx
            messagebox.showinfo("Mic selected", f"Selected device: {idx} - {names[idx]}")
        except Exception as e:
            messagebox.showerror("Invalid", "Please enter valid integer device index.")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose folder to save recordings", initialdir=str(BASE_DIR))
        if folder:
            global OUTPUT_FOLDER, TMP_VIDEO, TMP_AUDIO, OUTPUT_VIDEO
            OUTPUT_FOLDER = Path(folder).resolve()
            TMP_VIDEO = OUTPUT_FOLDER / "tmp_video.avi"
            TMP_AUDIO = OUTPUT_FOLDER / "tmp_audio.wav"
            OUTPUT_VIDEO = _timestamped_output_name(OUTPUT_FOLDER)
            self.out_path_label.configure(text=f"Save to: {OUTPUT_FOLDER}")

    def _on_minimize(self, event=None):
        # if minimized, still keep overlay visible if user wants overlay even when minimized
        try:
            st = str(self.state()) if hasattr(self, "state") else ""
            # if minimized and user still wants overlay visible, do nothing (keep visible)
            if (st == "iconic" or st == "withdrawn") and not self.show_overlay_var.get():
                try:
                    self.overlay_win.withdraw()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_restore(self, event=None):
        # restore overlay if enabled
        if self.show_overlay_var.get():
            try:
                if self._cam_thread and self._cam_thread.is_alive():
                    self.overlay_win.deiconify()
            except Exception:
                pass

    # ============== PREVIEW ==============
    def start_camera_preview(self):
        with _preview_lock:
            if self._cam_thread and self._cam_thread.is_alive():
                messagebox.showinfo("Preview", "Camera preview already running.")
                return

            cam_url = self.cam_url_var.get().strip() or None
            # parse cam_select_var to decide cam index
            cam_choice = self.cam_select_var.get()
            cam_index = None
            if cam_choice and cam_choice.startswith("Local index"):
                try:
                    cam_index = int(cam_choice.split()[-1])
                except Exception:
                    cam_index = None

            self._cam_stop_event = threading.Event()
            overlay_logo = None
            if self.logo_pil:
                try:
                    overlay_logo = self.logo_pil.copy().resize((64, 64), Image.Resampling.LANCZOS)
                except Exception:
                    overlay_logo = self.logo_pil

            self._cam_thread = threading.Thread(target=camera_capture_loop, args=(cam_url, self._cam_stop_event, self._preview_callback, overlay_logo, cam_index), daemon=True)
            self._cam_thread.start()
            if self.show_overlay_var.get():
                self.overlay_win.deiconify()
            self.status_var.set("Camera preview running")
            debug_print("Camera preview started.")

    def _preview_callback(self, pil_img):
        try:
            # update overlay preview image
            self.after(0, lambda img=pil_img: self.overlay_win.update_preview(img))
        except Exception as e:
            debug_print("_preview_callback error:", e)

    def stop_camera_preview(self):
        with _preview_lock:
            if self._cam_stop_event:
                self._cam_stop_event.set()
            if self._cam_thread:
                self._cam_thread.join(timeout=2.0)
            try:
                self.overlay_win.withdraw()
            except Exception:
                pass
            self.status_var.set("Preview stopped")
            debug_print("Camera preview stopped.")

    def _switch_webcam(self):
        # simple approach: stop preview, then choose next index and restart
        with _preview_lock:
            current = self.cam_select_var.get()
            # rotate through options
            values = self.cam_options
            try:
                idx = values.index(current)
                next_idx = (idx + 1) % len(values)
            except Exception:
                next_idx = 0
            self.cam_select_var.set(values[next_idx])
            # restart
            self.stop_camera_preview()
            time.sleep(0.15)
            self.start_camera_preview()
            debug_print("Switched webcam to:", values[next_idx])

    # ============== RECORDING ==============
    def start_recording(self):
        global _recording_flag, _selected_input_device, audio_queue, TMP_VIDEO, TMP_AUDIO, OUTPUT_VIDEO, _frames_written, _last_video_start_ts, _first_audio_ts, _paused_flag

        if _recording_flag:
            messagebox.showinfo("Recording", "Already recording.")
            return

        # ensure preview running for PiP
        if not (self._cam_thread and self._cam_thread.is_alive()):
            self.start_camera_preview()
            time.sleep(0.25)

        # output names (timestamped)
        OUTPUT_VIDEO = _timestamped_output_name(OUTPUT_FOLDER)
        TMP_VIDEO = OUTPUT_FOLDER / "tmp_video.avi"
        TMP_AUDIO = OUTPUT_FOLDER / "tmp_audio.wav"
        self.out_path_label.configure(text=f"Save to: {OUTPUT_FOLDER}")

        # remove old tmp files
        try:
            if TMP_VIDEO.exists():
                TMP_VIDEO.unlink()
            if TMP_AUDIO.exists():
                TMP_AUDIO.unlink()
        except Exception:
            pass

        # clear audio queue and counters
        audio_queue = queue.Queue()
        with _frames_written_lock:
            _frames_written = 0
        _last_video_start_ts = None
        _first_audio_ts = None
        _paused_flag = False
        self.btn_pause.configure(state="normal", text="⏸ Pause")

        # select device: if user selected explicit mic, use that, else auto
        if self.selected_mic_index is not None:
            _selected_input_device = self.selected_mic_index
        else:
            _selected_input_device = select_input_device(prefer_headset=True) if self.enable_audio else None

        debug_print("Selected input device:", _selected_input_device)

        # start audio capture thread
        self._audio_stop_event = threading.Event()
        self._audio_thread = threading.Thread(target=audio_capture_loop, args=(self._audio_stop_event, _selected_input_device, self.enable_audio), daemon=True)
        self._audio_thread.start()

        # start recording thread
        self._record_stop_event = threading.Event()
        mode = self.mode_var.get()
        self._record_thread = threading.Thread(target=recording_loop, args=(mode, self._record_stop_event, self.overlay_win.get_state, FPS, self.enable_audio), daemon=True)
        self._record_thread.start()

        _recording_flag = True
        self.status_var.set("Recording...")
        self.overlay_win.deiconify()
        self._record_start_time = time.time()
        # start timer update
        self._start_timer()
        debug_print("Recording started. Output:", OUTPUT_VIDEO)

    def _start_timer(self):
        self.elapsed_seconds = 0
        self._update_timer()

    def _update_timer(self):
        global _frames_written
        if not _recording_flag:
            return
        self.elapsed_seconds = int(time.time() - (self._record_start_time or time.time()))
        hh = self.elapsed_seconds // 3600
        mm = (self.elapsed_seconds % 3600) // 60
        ss = self.elapsed_seconds % 60
        with _frames_written_lock:
            frames = _frames_written
        fps_val = (frames / self.elapsed_seconds) if (self.elapsed_seconds > 0) else 0.0
        self.timer_var.set(f"Duration: {hh:02d}:{mm:02d}:{ss:02d} | FPS: {fps_val:.2f}")
        # update overlay too
        try:
            self.overlay_win.set_time(f"{hh:02d}:{mm:02d}:{ss:02d}")
            self.overlay_win.set_fps(fps_val)
        except Exception:
            pass
        # schedule next update
        self._timer_job = self.after(1000, self._update_timer)

    def _toggle_pause(self):
        global _paused_flag
        if not _recording_flag:
            return
        _paused_flag = not _paused_flag
        if _paused_flag:
            self.btn_pause.configure(text="▶ Resume")
            self.status_var.set("Paused")
        else:
            self.btn_pause.configure(text="⏸ Pause")
            self.status_var.set("Recording...")

    def stop_recording(self):
        global _recording_flag, audio_queue, _paused_flag
        if not _recording_flag:
            # treat as stop preview
            self.stop_camera_preview()
            return

        # signal threads to stop
        if self._record_stop_event:
            self._record_stop_event.set()
        if self._audio_stop_event:
            self._audio_stop_event.set()

        debug_print("Stopping recording threads; waiting for them to finish cleanly.")
        if self._record_thread:
            self._record_thread.join(timeout=8.0)
            debug_print("Record thread alive after join?:", self._record_thread.is_alive())
        if self._audio_thread:
            self._audio_thread.join(timeout=5.0)
            debug_print("Audio thread alive after join?:", self._audio_thread.is_alive())

        # clear audio queue as extra safety
        audio_queue = queue.Queue()

        _recording_flag = False
        _paused_flag = False
        self.btn_pause.configure(state="disabled", text="⏸ Pause")
        self.status_var.set("Merging audio & video...")
        # stop timer
        if self._timer_job:
            try:
                self.after_cancel(self._timer_job)
            except Exception:
                pass
            self._timer_job = None

        # run merge in background thread
        self._merge_thread = threading.Thread(target=self._merge_worker, daemon=True)
        self._merge_thread.start()

    def _merge_worker(self):
        """
        Background worker to merge TMP_VIDEO and TMP_AUDIO into OUTPUT_VIDEO.
        Also handles optional separate audio saving.
        Robustly waits until TMP_AUDIO/TMP_VIDEO exist (recording thread finalizer may write audio slightly later).
        Uses -itsoffset when audio started after video so video acts as master clock (Option A).
        """
        try:
            # Wait until TMP_VIDEO exists (writer closed) AND TMP_AUDIO exists (recording loop wrote it).
            waited = 0.0
            max_wait = 8.0  # give extra time for audio finalizer to write
            while waited < max_wait and (not TMP_VIDEO.exists() or not TMP_AUDIO.exists()):
                time.sleep(0.12)
                waited += 0.12
            if not TMP_VIDEO.exists() or not TMP_AUDIO.exists():
                debug_print("TMP files exist? video:", TMP_VIDEO.exists(), "audio:", TMP_AUDIO.exists())
                self.after(0, lambda: (self.status_var.set("Error: missing files"),
                                       messagebox.showwarning("Missing files",
                                                              f"Recording files missing.\nVideo exists: {TMP_VIDEO.exists()}\nAudio exists: {TMP_AUDIO.exists()}")) )
                return

            ff = ffmpeg_path()
            if not ff:
                self.after(0, lambda: messagebox.showerror("FFmpeg Missing", "FFmpeg not found. Place ffmpeg.exe in script folder or add ffmpeg to PATH."))
                self.after(0, lambda: self.status_var.set("Error: ffmpeg missing"))
                return

            # choose codec flags (default to libx264 for compatibility)
            use_gpu = False
            if use_gpu:
                codec_args = ["-c:v", "h264_nvenc", "-preset", "p7"]
            else:
                codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

            # Use audio/video offset if recorded by the capture loop
            
            global _audio_video_offset
            try:
                offset = float(_audio_video_offset)
            except Exception:
                offset -= 0.2
            # Build merge command - keep video as master; optionally shift audio with -itsoffset
            # If audio started later than video (offset > 0), we need to shift audio input forward
            
            if offset > 0.003:
                # use itsoffset before the audio input to shift it forward
                cmd = [ff, "-y", "-i", str(TMP_VIDEO), "-itsoffset", f"{offset:.3f}", "-i", str(TMP_AUDIO)] + codec_args + ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "aresample=async=1", "-vsync", "2", str(OUTPUT_VIDEO)]
            elif offset < -0.003:
                # audio started before video: trim audio with -ss (create trimmed audio first then merge)
                trimmed_audio = TMP_AUDIO.with_name(TMP_AUDIO.stem + "_trimmed.wav")
                trim_start = -offset  # remove front of audio
                tcmd = [ff, "-y", "-i", str(TMP_AUDIO), "-ss", f"{trim_start:.3f}", "-c", "copy", str(trimmed_audio)]
                debug_print("Trimming audio before merge:", " ".join(tcmd))
                tres = subprocess.run(tcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if tres.returncode != 0:
                    debug_print("Audio trim stderr:", tres.stderr.decode(errors="ignore"))
                    # fallback to using original audio
                    audio_input_path = str(TMP_AUDIO)
                else:
                    audio_input_path = str(trimmed_audio)
                cmd = [ff, "-y", "-i", str(TMP_VIDEO), "-i", audio_input_path] + codec_args + ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "aresample=async=1", "-vsync", "2", str(OUTPUT_VIDEO)]
            else:
                # No significant offset or unknown; merge normally but with audio resample/filter
                cmd = [ff, "-y", "-i", str(TMP_VIDEO), "-i", str(TMP_AUDIO)] + codec_args + [
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    "-af", "aresample=async=1:first_pts=0",
                    "-vsync", "cfr",
                str(OUTPUT_VIDEO)
            ]

            debug_print("Running merge cmd:", " ".join(cmd))
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                debug_print("FFmpeg merge stderr:", res.stderr.decode(errors="ignore"))
                self.after(0, lambda: (self.status_var.set("Error merging"), messagebox.showerror("Merge failed", res.stderr.decode(errors="ignore"))))
                return

            # optionally save separate audio as WAV (user requested)
            if self.save_separate_audio_var.get():
                out_audio = OUTPUT_FOLDER / (OUTPUT_VIDEO.stem + "_audio.wav")
                try:
                    with open(TMP_AUDIO, "rb") as fsrc, open(out_audio, "wb") as fdst:
                        fdst.write(fsrc.read())
                except Exception as e:
                    debug_print("Failed to copy separate audio:", e)

            # cleanup tmp files and temporary trimmed audio if any
            try:
                TMP_VIDEO.unlink(missing_ok=True)
                TMP_AUDIO.unlink(missing_ok=True)
                if 'trimmed_audio' in locals() and Path(trimmed_audio).exists():
                    Path(trimmed_audio).unlink(missing_ok=True)
            except Exception:
                pass

            # notify UI
            self.after(0, lambda: (self.status_var.set(f"Saved: {OUTPUT_VIDEO.name}"), messagebox.showinfo("Done", f"Recording saved:\n{OUTPUT_VIDEO}")))
        except Exception as e:
            debug_print("Merge worker exception:", e)
            self.after(0, lambda: (self.status_var.set("Error merging"), messagebox.showerror("Error merging", str(e))))

    # ============== TRIM UI ==============
    def trim_video_ui(self):
        """Opens a simple FFmpeg-based trim dialog."""
        try:
            last_file = filedialog.askopenfilename(
                title="Select MP4 to trim",
                initialdir=str(OUTPUT_FOLDER),
                filetypes=[("MP4 videos", "*.mp4"), ("All files", "*.*")]
            )
            if not last_file:
                return

            # Ask for trim start and end times
            start = simpledialog.askfloat("Trim", "Start time (in seconds):", minvalue=0.0, initialvalue=0.0)
            if start is None:
                return
            end = simpledialog.askfloat("Trim", "End time (in seconds):", minvalue=start + 0.1)
            if end is None or end <= start:
                messagebox.showerror("Trim Error", "End time must be greater than start time.")
                return

            # Generate output name
            out_path = Path(last_file).with_name(Path(last_file).stem + f"_trim_{int(start)}_{int(end)}.mp4")

            self.status_var.set("Trimming video (background)...")

            # Run trim worker in background thread
            threading.Thread(target=self._trim_worker, args=(Path(last_file), float(start), float(end), out_path), daemon=True).start()

        except Exception as e:
            debug_print("Trim UI error:", e)
            messagebox.showerror("Trim Feature", f"Unexpected error: {e}")

    def _trim_worker(self, in_path: Path, start: float, end: float, out_path: Path):
        """Performs the actual trim operation using FFmpeg in background."""
        try:
            ff = ffmpeg_path()
            if not ff:
                self.after(0, lambda: messagebox.showerror("FFmpeg Missing", "FFmpeg not found in PATH or script folder."))
                return

            dur = end - start
            cmd = [
                ff, "-y",
                "-i", str(in_path),
                "-ss", f"{start:.3f}",
                "-t", f"{dur:.3f}",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(out_path)
            ]

            debug_print("Running trim cmd:", " ".join(cmd))
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if res.returncode != 0:
                err_text = res.stderr.decode(errors="ignore")
                debug_print("Trim stderr:", err_text)
                self.after(0, lambda: messagebox.showerror("Trim Failed", f"Error during trimming:\n\n{err_text}"))
                self.after(0, lambda: self.status_var.set("Trim failed"))
                return

            self.after(0, lambda: (messagebox.showinfo("Trim Completed", f"Trimmed file saved successfully:\n{out_path}"), self.status_var.set("Ready")))
        except Exception as e:
            debug_print("Trim worker exception:", e)
            self.after(0, lambda: messagebox.showerror("Trim Worker Error", str(e)))

    # ============== CLOSE ==============
    def on_close(self):
        debug_print("Closing app; stopping threads if any.")
        try:
            if self._cam_stop_event:
                self._cam_stop_event.set()
        except Exception:
            pass
        try:
            if self._cam_thread:
                self._cam_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._record_stop_event:
                self._record_stop_event.set()
        except Exception:
            pass
        try:
            if self._record_thread:
                self._record_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._audio_stop_event:
                self._audio_stop_event.set()
        except Exception:
            pass
        try:
            if self._audio_thread:
                self._audio_thread.join(timeout=2.0)
        except Exception:
            pass
        # small sleep for safe cleanup
        time.sleep(0.2)
        try:
            self.destroy()
        except Exception:
            sys.exit(0)

# ================ MAIN ================
def main():
    ensure_dirs()
    app = RoyalRecorderApp()
    app.mainloop()

if __name__ == "__main__":
    main()
