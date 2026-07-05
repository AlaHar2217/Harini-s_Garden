"""
AI-Based Virtual Rose Plant Growth Using Hand Gesture Recognition
-------------------------------------------------------------------
Interactive application that uses a webcam to detect hand gestures and
grow a small garden of cute, cartoon-style roses.

Controls
    Open hand   -> grow the garden
    Closed fist -> shrink the garden
    q           -> quit

Requires: opencv-python, mediapipe, numpy
    pip install opencv-python mediapipe numpy
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np


# ============================================================================
# Configuration — every tunable constant lives here so behaviour and look
# can be adjusted without hunting through the drawing code.
# ============================================================================

class Colors:
    """BGR colors (OpenCV order), grouped by what they're used for."""
    SOIL = (60, 90, 140)
    STEM = (70, 160, 90)
    LEAF = (90, 200, 120)
    LEAF_HIGHLIGHT = (140, 225, 170)

    BUD = (150, 130, 235)
    BUD_HIGHLIGHT = (210, 200, 250)

    PETAL_OUTER = (150, 140, 235)     # soft pink
    PETAL_MID = (170, 160, 245)       # candy pink
    PETAL_INNER = (200, 195, 250)     # pale blush
    PETAL_HIGHLIGHT = (255, 255, 255)

    CENTER = (110, 210, 255)          # warm cream-yellow
    FACE = (60, 60, 90)
    BLUSH = (170, 170, 250)
    SPARKLE = (255, 250, 240)

    GRASS_LIGHT = (90, 170, 90)
    GRASS_DARK = (50, 120, 60)
    TITLE_TEXT = (170, 120, 255)


STAGE_COUNT = 7                        # growth goes from 0 (seed) to 7 (full bloom)
STAGE_HEIGHTS = [0, 45, 90, 135, 180, 220, 250, 275]
GROWTH_STEP = 0.10                     # per-frame growth while hand is open
SHRINK_STEP = 0.15                     # per-frame shrink while fist is closed
GROWTH_EASE = 0.18                     # how quickly growth eases toward its target

PINCH_OPEN_THRESHOLD = 0.08
SPREAD_OPEN_THRESHOLD = 0.20
PINCH_CLOSED_THRESHOLD = 0.08
SPREAD_CLOSED_THRESHOLD = 0.18
GESTURE_SMOOTHING_FRAMES = 5            # rolling average window to stop jitter

NUM_ROSES = 7
TARGET_FPS = 60


# ============================================================================
# RosePlant — now with a friendlier, kawaii-style bloom
# ============================================================================

@dataclass
class RosePlant:
    """A single virtual rose that grows in discrete stages:

    0        1        2             5          6           7
    (soil) - sprout - leafy stem - budding - blooming - full bloom
    """
    base_x: int
    base_y: int
    growth_level: float = 0.0
    target_growth: float = 0.0
    phase: float = field(default_factory=lambda: np.random.uniform(0, 2 * math.pi))

    def set_target(self, target: float) -> None:
        self.target_growth = max(0.0, min(float(STAGE_COUNT), target))

    def update(self, dt_frames: float = 1.0) -> None:
        """Smoothly ease growth_level toward target_growth (a gentle bounce
        instead of a linear snap makes the garden feel alive)."""
        self.growth_level += (self.target_growth - self.growth_level) * GROWTH_EASE * dt_frames
        self.phase += 0.06

    # -- drawing -------------------------------------------------------

    def draw(self, frame: np.ndarray) -> None:
        stage = int(self.growth_level)
        progress = self.growth_level - stage

        self._draw_soil(frame)

        if stage >= 1:
            self._draw_stem(frame, stage, progress)
        if stage >= 2:
            self._draw_leaves(frame, stage, progress)
        if stage >= 5:
            self._draw_bud(frame, stage, progress)
        if stage >= 6:
            self._draw_bloom(frame, stage, progress)

    def _stem_height(self, stage: int, progress: float) -> float:
        h = STAGE_HEIGHTS[min(stage, STAGE_COUNT)]
        if stage < STAGE_COUNT:
            h += progress * (STAGE_HEIGHTS[min(stage + 1, STAGE_COUNT)] - h)
        return h

    def _draw_soil(self, frame: np.ndarray) -> None:
        cv2.ellipse(frame, (self.base_x, self.base_y), (78, 18), 0, 0, 180,
                    Colors.SOIL, -1)

    def _draw_stem(self, frame: np.ndarray, stage: int, progress: float) -> None:
        stem_height = self._stem_height(stage, progress)
        sway = math.sin(self.phase) * 3 if stage >= 6 else 0
        thickness = min(4 + stage // 2, 6)

        top = (int(self.base_x + sway), int(self.base_y - stem_height))
        # a slight curve reads as friendlier than a perfectly straight line
        mid = (int(self.base_x + sway * 0.5), int(self.base_y - stem_height * 0.5))

        pts = np.array([[self.base_x, self.base_y], mid, top], dtype=np.int32)
        cv2.polylines(frame, [pts], False, Colors.STEM, thickness, cv2.LINE_AA)

    def _draw_leaves(self, frame: np.ndarray, stage: int, progress: float) -> None:
        num_leaves = min((stage - 1) * 2, 8)
        stem_height = self._stem_height(stage, progress)

        for i in range(num_leaves):
            leaf_y = int(self.base_y - stem_height * (i + 1) / (num_leaves + 2))
            side = 1 if i % 2 == 0 else -1
            wobble = math.sin(self.phase + i) * 2
            leaf_x = int(self.base_x + side * 24 + wobble)

            length = 14 + stage * 2
            width = 9 + stage
            angle = side * 40

            # rounded, heart-ish leaf made of two overlapping lobes
            cv2.ellipse(frame, (leaf_x, leaf_y), (length, width), angle, 0, 360,
                        Colors.LEAF, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (leaf_x - side * 4, leaf_y - 3), (length // 2, width // 2),
                        angle, 0, 360, Colors.LEAF_HIGHLIGHT, -1, cv2.LINE_AA)

    def _draw_bud(self, frame: np.ndarray, stage: int, progress: float) -> None:
        if stage != 5:
            return
        stem_height = self._stem_height(stage, progress)
        x = self.base_x
        y = int(self.base_y - stem_height)
        size = int(9 + progress * 7)

        cv2.circle(frame, (x, y), size, Colors.BUD, -1, cv2.LINE_AA)
        cv2.circle(frame, (x - size // 3, y - size // 3), max(2, size // 3),
                   Colors.BUD_HIGHLIGHT, -1, cv2.LINE_AA)

    def _draw_bloom(self, frame: np.ndarray, stage: int, progress: float) -> None:
        stem_height = self._stem_height(stage, progress)
        cx = self.base_x
        cy = int(self.base_y - stem_height)

        if stage == 6:
            size = 34 + progress * 30
        else:
            size = 64 + math.sin(self.phase) * 2  # gentle "breathing"

        self._petal_ring(frame, cx, cy, size, count=7, radius_f=0.55,
                          rx_f=0.42, ry_f=0.34, angle_offset=0, color=Colors.PETAL_OUTER)
        self._petal_ring(frame, cx, cy, size, count=6, radius_f=0.36,
                          rx_f=0.36, ry_f=0.29, angle_offset=25, color=Colors.PETAL_MID)
        self._petal_ring(frame, cx, cy, size, count=5, radius_f=0.18,
                          rx_f=0.28, ry_f=0.24, angle_offset=10, color=Colors.PETAL_INNER)

        # center
        center_r = int(size * 0.22)
        cv2.circle(frame, (cx, cy), center_r, Colors.CENTER, -1, cv2.LINE_AA)

        if stage >= 6:
            self._draw_cute_face(frame, cx, cy, center_r)

        if stage == STAGE_COUNT:
            self._draw_sparkles(frame, cx, cy, size)

    def _petal_ring(self, frame, cx, cy, size, count, radius_f, rx_f, ry_f,
                     angle_offset, color) -> None:
        for i in range(count):
            angle = (360 / count) * i + angle_offset
            rad = math.radians(angle)
            px = int(cx + math.cos(rad) * size * radius_f)
            py = int(cy + math.sin(rad) * size * radius_f)
            rx, ry = int(size * rx_f), int(size * ry_f)

            # rounded petal: a soft ellipse plus a small round highlight
            cv2.ellipse(frame, (px, py), (rx, ry), angle, 0, 360, color, -1, cv2.LINE_AA)
            hl = (int(px - math.cos(rad) * rx * 0.25), int(py - math.sin(rad) * ry * 0.25))
            cv2.circle(frame, hl, max(2, rx // 5), Colors.PETAL_HIGHLIGHT, -1, cv2.LINE_AA)

    def _draw_cute_face(self, frame: np.ndarray, cx: int, cy: int, center_r: int) -> None:
        """A tiny kawaii face on the fully-open bloom: two round eyes,
        a small smile, and blush circles."""
        eye_r = max(2, center_r // 6)
        eye_dx = center_r // 2
        eye_y = cy - center_r // 6

        cv2.circle(frame, (cx - eye_dx, eye_y), eye_r, Colors.FACE, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + eye_dx, eye_y), eye_r, Colors.FACE, -1, cv2.LINE_AA)

        smile_box = (cx - center_r // 3, cy + center_r // 8)
        cv2.ellipse(frame, smile_box, (center_r // 3, center_r // 4), 0, 20, 160,
                    Colors.FACE, max(1, eye_r // 2), cv2.LINE_AA)

        blush_r = max(2, center_r // 5)
        cv2.circle(frame, (cx - eye_dx - blush_r, cy + center_r // 6), blush_r,
                   Colors.BLUSH, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + eye_dx + blush_r, cy + center_r // 6), blush_r,
                   Colors.BLUSH, -1, cv2.LINE_AA)

    def _draw_sparkles(self, frame: np.ndarray, cx: int, cy: int, size: float) -> None:
        """A few twinkling plus-shaped sparkles orbiting a fully bloomed rose."""
        for i in range(4):
            angle = self.phase * 1.5 + i * (math.pi / 2)
            dist = size * 0.9
            sx = int(cx + math.cos(angle) * dist)
            sy = int(cy + math.sin(angle) * dist * 0.6 - size * 0.3)
            twinkle = (math.sin(self.phase * 3 + i) + 1) / 2
            if twinkle < 0.35:
                continue
            r = int(3 + twinkle * 3)
            cv2.line(frame, (sx - r, sy), (sx + r, sy), Colors.SPARKLE, 1, cv2.LINE_AA)
            cv2.line(frame, (sx, sy - r), (sx, sy + r), Colors.SPARKLE, 1, cv2.LINE_AA)

    def get_growth_percentage(self) -> float:
        return (self.growth_level / STAGE_COUNT) * 100


# ============================================================================
# GestureDetector — same open/closed detection, now with rolling smoothing
# so a single noisy frame can't flip the command back and forth.
# ============================================================================

class GestureDetector:
    """Detects a hand and classifies it as open, closed, or idle."""

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self._distance_history: deque[float] = deque(maxlen=GESTURE_SMOOTHING_FRAMES)
        self._spread_history: deque[float] = deque(maxlen=GESTURE_SMOOTHING_FRAMES)

    def detect_hand(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame: np.ndarray, results) -> None:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def calculate_gesture_value(self, results) -> tuple[str, str]:
        """Returns (gesture_label, command) where command is 'grow', 'shrink' or 'idle'."""
        if not results.multi_hand_landmarks:
            self._distance_history.clear()
            self._spread_history.clear()
            return "No Hand Detected", "idle"

        landmarks = results.multi_hand_landmarks[0].landmark
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        avg_finger_y = (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4
        finger_spread = abs(wrist.y - avg_finger_y)

        self._distance_history.append(distance)
        self._spread_history.append(finger_spread)
        smooth_distance = sum(self._distance_history) / len(self._distance_history)
        smooth_spread = sum(self._spread_history) / len(self._spread_history)

        is_open = smooth_distance > PINCH_OPEN_THRESHOLD or smooth_spread > SPREAD_OPEN_THRESHOLD
        is_closed = smooth_distance < PINCH_CLOSED_THRESHOLD and smooth_spread < SPREAD_CLOSED_THRESHOLD

        if is_open:
            return "Open Hand (Growing)", "grow"
        if is_closed:
            return "Closed Fist (Shrinking)", "shrink"
        return "Idle", "idle"

    def release(self) -> None:
        self.hands.close()


# ============================================================================
# VirtualRoseApp — main loop
# ============================================================================

class VirtualRoseApp:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise RuntimeError("Could not open the webcam. Is it connected and not in use elsewhere?")

        self.gesture_detector = GestureDetector()

        spacing = self.width // (NUM_ROSES + 1)
        self.rose_plants = [
            RosePlant(
                base_x=spacing * (i + 1),
                base_y=self.height - 50 + (i % 3 - 1) * 10,
            )
            for i in range(NUM_ROSES)
        ]

        self.prev_time = time.time()
        self.fps = 0.0

    # -- drawing ---------------------------------------------------------

    def _draw_background(self, frame: np.ndarray) -> None:
        grass_height = 100
        cv2.rectangle(frame, (0, self.height - grass_height),
                      (self.width, self.height), Colors.GRASS_LIGHT, -1)
        for i in range(0, self.width, 30):
            cv2.line(frame, (i, self.height - grass_height), (i + 15, self.height),
                     Colors.GRASS_DARK, 1)

    def _draw_ui(self, frame: np.ndarray, gesture_type: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Virtual Rose Garden"
        text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, 50), font, 1.2, Colors.TITLE_TEXT, 3, cv2.LINE_AA)

        cv2.putText(frame, gesture_type, (20, self.height - 20), font, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        print("=" * 60)
        print("    Virtual Rose Garden - Hand Gesture Control")
        print("=" * 60)
        print("\nControls:")
        print("  Open Hand  -> Grow all roses")
        print("  Close Fist -> Shrink all roses")
        print("  Press 'q'  -> Quit application")
        print("\n" + "=" * 60)

        target_frame_time = 1.0 / TARGET_FPS

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: failed to read from webcam, stopping.")
                break

            frame = cv2.flip(frame, 1)
            self._draw_background(frame)

            results = self.gesture_detector.detect_hand(frame)
            gesture_type, command = self.gesture_detector.calculate_gesture_value(results)

            for rose in self.rose_plants:
                if command == "grow":
                    rose.set_target(rose.target_growth + GROWTH_STEP * 4)
                elif command == "shrink":
                    rose.set_target(rose.target_growth - SHRINK_STEP * 4)
                rose.update()
                rose.draw(frame)

            self.gesture_detector.draw_landmarks(frame, results)
            self._draw_ui(frame, gesture_type)

            now = time.time()
            frame_time = now - self.prev_time
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
            now = time.time()
            self.fps = 1.0 / (now - self.prev_time) if now > self.prev_time else self.fps
            self.prev_time = now

            cv2.imshow("Virtual Rose Garden - Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self) -> None:
        self.cap.release()
        self.gesture_detector.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def main() -> None:
    try:
        app = VirtualRoseApp()
        app.run()
    except Exception as exc:
        print(f"Error: {exc}")
        print("\nMake sure you have installed the required packages:")
        print("pip install opencv-python mediapipe numpy")


if __name__ == "__main__":
    main()