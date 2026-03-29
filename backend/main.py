"""
main.py — Shape Detector, okno testowe (samo OpenCV, bez Electrona)

Uruchomienie:
  python main.py                     # wbudowana kamerka (indeks 0)
  python main.py --source 1          # druga kamerka / DroidCam Client
  python main.py --source http://IP:8080/video   # IP Webcam (Android)

Sterowanie:
  Q  — wyjście
  S  — zapis aktualnej klatki jako PNG
  V  — cyklowanie trybów widoku: Normal → Debug (maski) → Canny → ...
  P  — pauza / wznowienie
"""

import argparse
import time

import cv2
import numpy as np

from detector import (build_canny_view, build_debug_overlay,
                      detect_shapes, draw_detections)

# ---------------------------------------------------------------------------
# Tryby podglądu
# ---------------------------------------------------------------------------
VIEW_NORMAL = 0   # czysty obraz + kontury detekcji
VIEW_DEBUG  = 1   # nakładka masek kolorów + dodatkowe metryki
VIEW_CANNY  = 2   # krawędzie Canny (białe linie) + kontury detekcji

_VIEW_NAMES = {
    VIEW_NORMAL: "Normal",
    VIEW_DEBUG:  "Debug (masks)",
    VIEW_CANNY:  "Canny edges",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shape Detector")
    parser.add_argument(
        "--source", default="0",
        help="Źródło wideo: 0/1 (indeks kamerki) lub URL (IP Webcam)"
    )
    return parser.parse_args()


def list_cameras(max_index: int = 5) -> list[int]:
    """Skanuje indeksy 0–max_index i zwraca listę działających kamer."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def open_capture(source: str) -> cv2.VideoCapture:
    """
    Otwiera strumień wideo. Obsługuje:
      - indeks całkowity (np. "0", "1") — wbudowana kamera lub wirtualna (DroidCam Client)
      - URL HTTP MJPEG — np. IP Webcam: http://IP:8080/video

    Uwaga dot. DroidCam v6+: port 4747 to panel sterowania (HTML), nie strumień wideo.
    Aby użyć DroidCam przez WiFi zainstaluj klienta Windows z dev47apps.com —
    telefon pojawi się jako wirtualna kamera (indeks 1 lub 2).
    Alternatywnie użyj aplikacji 'IP Webcam' (Pavel Khlebovich) — wystawia MJPEG na porcie 8080.
    """
    if source.isdigit():
        idx = int(source)
        # MSMF (Microsoft Media Foundation) — domyślny backend Windows,
        # działa z wirtualnymi kamerami (DroidCam Client, OBS Virtual Camera)
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if not cap.isOpened():
            # Fallback: bez jawnego backendu (OpenCV sam wybierze)
            cap = cv2.VideoCapture(idx)
    else:
        # Strumień HTTP — sprawdź najpierw czy serwer odpowiada i co zwraca
        import urllib.request
        try:
            r = urllib.request.urlopen(source, timeout=4)
            content_type = r.getheader("Content-Type", "")
            r.close()
            if "text/html" in content_type:
                print(
                    f"[WARN] '{source}' zwraca stronę HTML, nie strumień wideo.\n"
                    f"       DroidCam v6+: zainstaluj klienta Windows i użyj --source 1\n"
                    f"       IP Webcam: użyj http://IP:8080/video"
                )
        except Exception as e:
            print(f"[WARN] Nie można połączyć z '{source}': {e}")

        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        available = list_cameras()
        hint = f"\n  Dostępne kamerki (indeksy): {available if available else 'brak'}"
        if not source.isdigit():
            hint += (
                "\n  Wskazówki:\n"
                "  1. DroidCam v6+: zainstaluj klienta Windows (dev47apps.com) i użyj --source 1\n"
                "  2. IP Webcam (Android): http://IP:8080/video\n"
                "  3. Sprawdź numer indeksu kamerki: --source 0, --source 1, --source 2"
            )
        raise RuntimeError(f"Cannot open video source: {source}{hint}")

    # Ustawienie rozdzielczości — kamera może zignorować dla źródeł sieciowych
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap


def draw_hud(frame: np.ndarray, fps: float, detections: list,
             view_mode: int, paused: bool) -> None:
    """Rysuje nakładkę HUD: FPS, liczba detekcji, tryb widoku, skróty klawiszowe."""
    h, w = frame.shape[:2]

    # FPS i liczba wykrytych obiektów — lewy górny róg
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detected: {len(detections)}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

    # Nazwa aktywnego trybu — pod licznikiem detekcji
    mode_name = _VIEW_NAMES.get(view_mode, "?")
    mode_color = {VIEW_NORMAL: (200, 200, 200),
                  VIEW_DEBUG:  (0, 200, 255),
                  VIEW_CANNY:  (255, 200, 0)}.get(view_mode, (200, 200, 200))
    cv2.putText(frame, mode_name,
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

    # Legenda masek kolorów widoczna tylko w trybie Debug
    if view_mode == VIEW_DEBUG:
        legend = [
            ("red",    (0,   0,   200)),
            ("green",  (0,   200, 0)),
            ("blue",   (200, 0,   0)),
            ("yellow", (0,   200, 200)),
            ("orange", (0,   100, 220)),
            ("maroon", (60,  0,   140)),
            ("dark",   (120, 120, 120)),
            ("skin",   (80,  200, 255)),
        ]
        for i, (name, bgr) in enumerate(legend):
            y = 108 + i * 22
            cv2.rectangle(frame, (12, y - 12), (26, y + 2), bgr, -1)
            cv2.putText(frame, name, (32, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3, cv2.LINE_AA)

    # Skróty klawiszowe — prawy dolny róg
    hints = ["Q - quit", "S - save", "V - view", "P - pause"]
    for i, hint in enumerate(reversed(hints)):
        cv2.putText(frame, hint,
                    (w - 160, h - 12 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    cap = open_capture(args.source)

    print(f"[Shape Detector] Source: {args.source}")
    print("[Shape Detector] Press Q to quit. V to cycle views.")

    view_mode = VIEW_NORMAL
    paused = False
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    last_frame: np.ndarray | None = None
    last_debug_masks: dict = {}

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame — check video source.")
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        # Przelicz FPS co 30 klatek
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps = 30 / elapsed if elapsed > 0 else 0.0
            fps_timer = time.time()

        # --- Detekcja ---
        detections, debug_masks = detect_shapes(frame)
        if not paused:
            last_debug_masks = debug_masks

        # --- Wizualizacja zależna od trybu ---
        if view_mode == VIEW_DEBUG:
            # Nakładka kolorowych masek + szczegółowe etykiety z metrykami
            display = build_debug_overlay(frame, last_debug_masks)
            display = draw_detections(display, detections, debug=True)

        elif view_mode == VIEW_CANNY:
            # Białe krawędzie Canny na przyciemnionym obrazie + kontury detekcji
            canny = last_debug_masks.get("canny",
                                         np.zeros(frame.shape[:2], dtype=np.uint8))
            display = build_canny_view(frame, canny)
            display = draw_detections(display, detections)

        else:  # VIEW_NORMAL
            display = draw_detections(frame, detections)

        draw_hud(display, fps, detections, view_mode, paused)
        cv2.imshow("Shape Detector — press Q to quit", display)

        # --- Klawisze ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[Shape Detector] Quitting...")
            break

        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"[Shape Detector] Saved: {fname}")

        elif key == ord('v') or key == ord('d'):
            # V cykluje tryby widoku; D zachowany dla kompatybilności
            view_mode = (view_mode + 1) % 3
            print(f"[Shape Detector] View: {_VIEW_NAMES[view_mode]}")

        elif key == ord('p'):
            paused = not paused
            print(f"[Shape Detector] {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
