"""Live webcam recording: pipes processed frames + microphone audio into a
single ffmpeg process, producing one muxed video file.

Only the Linux audio path (PulseAudio, `-f pulse -i default`) is verified
working. Windows (DirectShow) and macOS (AVFoundation) branches are
best-effort placeholders for later — verify on those platforms before
relying on them; if the audio device can't be opened, recording falls back
to video-only automatically.
"""

import os
import platform
import subprocess
import tempfile
import threading
import time
from typing import List, Optional

import numpy as np


def _audio_input_args() -> List[str]:
    """ffmpeg args selecting the system default microphone as a second input.

    Empty list means "no audio input" (caller should fall back to video-only).
    """
    system = platform.system()
    if system == "Linux":
        # Verified: PulseAudio (or PipeWire's Pulse-compatible shim, which
        # covers most modern distros) exposes the default input as "default".
        return ["-f", "pulse", "-i", "default"]
    if system == "Windows":
        # NOT verified on real hardware. DirectShow needs an exact device
        # NAME, not "default" — enumerate with:
        #   ffmpeg -list_devices true -f dshow -i dummy
        # "Microphone" is a common but unreliable label and may need the
        # real device name substituted once tested on Windows.
        return ["-f", "dshow", "-i", "audio=Microphone"]
    if system == "Darwin":
        # NOT verified on real hardware. AVFoundation's default-mic index
        # varies by Mac; ":0" is common but unguaranteed. Enumerate with:
        #   ffmpeg -f avfoundation -list_devices true -i ""
        return ["-f", "avfoundation", "-i", ":0"]
    return []


def _audio_device_available(timeout: float = 2.0) -> bool:
    """Synchronously probe whether the mic input actually opens.

    A real device-open failure (missing PulseAudio, bad ALSA index, no
    permission, etc.) does not necessarily surface within a fixed short
    window after Popen — verified experimentally: an invalid ALSA device
    kept the recording process alive for over 0.3s before erroring, so a
    "sleep briefly then poll()" check on the *real* recording process raced
    and reported success right before the process died mid-recording,
    losing the capture. Running a short, separate, blocking capture-to-null
    test up front avoids that race and never touches real recorded frames.
    """
    audio_args = _audio_input_args()
    if not audio_args:
        return False
    test_cmd = (
        ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        + audio_args
        + ["-t", "0.3", "-f", "null", "-"]
    )
    try:
        result = subprocess.run(
            test_cmd, capture_output=True, timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


class LiveRecorder:
    """Encodes piped-in BGR frames (+ system mic audio) to a temp video file.

    write_frame() is meant to be called from the frame-processing thread,
    never the Qt UI thread — a stalled/slow encoder pipe must not be able to
    block the UI event loop.
    """

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.audio_enabled = False
        self._process: Optional[subprocess.Popen] = None
        self._temp_path: Optional[str] = None
        self._lock = threading.Lock()
        self._active = False

    def start(self) -> bool:
        """Start the encoder. Tries with mic audio first, falls back to
        video-only if the audio device can't be opened (e.g. no PulseAudio
        running). Returns False if even the video-only fallback fails."""
        self._temp_path = os.path.join(
            tempfile.gettempdir(), f"dlc_recording_{int(time.time())}.mp4"
        )
        if _audio_device_available() and self._try_start(with_audio=True):
            return True
        return self._try_start(with_audio=False)

    def _try_start(self, with_audio: bool) -> bool:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            # Timestamp each incoming raw frame by real arrival time rather
            # than assuming a fixed 1/fps spacing. The live processing
            # pipeline (detect+swap+enhance) frequently can't keep up with
            # the camera's nominal fps, so frames arrive slower/irregularly;
            # without this, ffmpeg still declares each frame as exactly
            # 1/fps long, understating the recording's real duration and
            # making playback look sped up (verified: 24 frames actually
            # spread over 3.03s wall-clock encoded as a 0.96s video).
            "-use_wallclock_as_timestamps", "1",
            "-i", "-",
        ]
        audio_args = _audio_input_args() if with_audio else []
        cmd += audio_args
        cmd += [
            # Output-side fixed framerate: conforms the wallclock-timestamped,
            # irregularly-paced input to a steady fps by duplicating frames
            # during slow stretches — this is what makes the encoded
            # duration match real elapsed time instead of frame-count/fps.
            "-r", str(self.fps),
            "-vsync", "cfr",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
        if audio_args:
            # -shortest: the mic input never EOFs on its own (it's a live
            # device), so without this ffmpeg would keep encoding audio-only
            # after the video pipe closes at stop() instead of finishing.
            cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [self._temp_path]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"[LiveRecorder] Failed to start ffmpeg: {e}")
            return False

        # Give ffmpeg a moment to fail fast (e.g. bad output path/codec)
        # before committing to this process. Audio-device availability is
        # NOT decided here — see _audio_device_available()'s docstring for
        # why a poll-after-sleep race is unreliable for that specifically.
        time.sleep(0.3)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            print(
                f"[LiveRecorder] ffmpeg exited immediately "
                f"(with_audio={with_audio}): {stderr.strip()}"
            )
            return False

        self._process = proc
        self.audio_enabled = with_audio
        self._active = True
        return True

    def write_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._active or self._process is None or self._process.stdin is None:
                return
            try:
                self._process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as e:
                print(f"[LiveRecorder] Write failed, stopping recording: {e}")
                self._active = False

    def stop(self) -> Optional[str]:
        """Stop encoding and return the finished temp file path, or None on
        failure (nothing usable was produced)."""
        with self._lock:
            self._active = False
            proc = self._process
            self._process = None

        if proc is None:
            return None

        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception as e:
            print(f"[LiveRecorder] Error closing ffmpeg: {e}")
            try:
                proc.kill()
            except Exception:
                pass
            return None

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            print(f"[LiveRecorder] ffmpeg exited with error: {stderr.strip()}")

        if (
            self._temp_path
            and os.path.isfile(self._temp_path)
            and os.path.getsize(self._temp_path) > 0
        ):
            return self._temp_path
        return None
