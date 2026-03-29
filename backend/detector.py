import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Zakresy kolorów w przestrzeni HSV — format: nazwa -> [(dolna, górna), ...]
# Czerwony ma dwa zakresy bo zawija się w kole barw (0° i 360°)
# Zielony: wyższe S (120) i cap na V (200) eliminują odbicia ekranu laptopa
# Bordowy (maroon): ciemny czerwony, niskie V (15-90)
# ---------------------------------------------------------------------------
COLOR_RANGES: dict[str, list[tuple]] = {
    "red":    [(np.array([0,   100, 70]),  np.array([8,   255, 220])),
               (np.array([165, 100, 70]),  np.array([180, 255, 220]))],
    "green":  [(np.array([45,  120, 70]),  np.array([78,  255, 200]))],
    "blue":   [(np.array([100, 90,  50]),  np.array([130, 255, 220]))],
    "yellow": [(np.array([22,  100, 100]), np.array([38,  255, 230]))],
    "orange": [(np.array([10,  160, 130]), np.array([15,  255, 230]))],
    "maroon": [(np.array([0,   100, 15]),  np.array([10,  255, 90])),
               (np.array([165, 100, 15]),  np.array([180, 255, 90]))],
}

# Kolory rysowania konturów (BGR)
DRAW_COLORS: dict[str, tuple] = {
    "red":     (0,   0,   220),
    "green":   (0,   200, 0),
    "blue":    (220, 80,  0),
    "yellow":  (0,   200, 220),
    "orange":  (0,   140, 255),
    "maroon":  (60,  0,   160),
    "dark":    (110, 110, 110),
    "unknown": (180, 180, 180),
}

# Kolory nakładki w trybie debug (BGR)
_DEBUG_MASK_COLORS: dict[str, tuple] = {
    "red":    (0,   0,   180),
    "green":  (0,   180, 0),
    "blue":   (180, 0,   0),
    "yellow": (0,   180, 180),
    "orange": (0,   100, 220),
    "maroon": (60,  0,   140),
    "dark":   (120, 120, 120),
    "skin":   (80,  180, 255),
}

# ---------------------------------------------------------------------------
# Progi filtrów kształtu
# ---------------------------------------------------------------------------
MIN_AREA         = 4000   # minimalna powierzchnia konturu w pikselach
MIN_SOLIDITY     = 0.62   # stosunek area/hull_area — odrzuca nieregularne kształty
MAX_ASPECT_RATIO = 5.0    # max proporcja dłuższej/krótszej krawędzi — odrzuca półki
MAX_SKIN_RATIO   = 0.40   # max udział pikseli skóry wewnątrz konturu (dłoń)
MAX_COMPACTNESS  = 3.5    # obwód²/(4π×pole): koło=1.0, nieregularna plama >> 3.5
MIN_BBOX_FILL    = 0.28   # min. stosunek pola konturu do prostokąta otaczającego

# Ciemne obiekty: wnętrze konturu musi być ciemniejsze niż X*średnia jasność klatki
DARK_BRIGHTNESS_RATIO = 0.55


def _build_skin_mask(hsv: np.ndarray) -> np.ndarray:
    """Maska typowych odcieni skóry — do odrzucania dłoni trzymających obiekty."""
    return cv2.inRange(hsv, np.array([0, 20, 60]), np.array([20, 110, 230]))


def classify_shape(contour) -> tuple[str, float]:
    """
    Klasyfikuje kształt konturu. Zwraca (nazwa, pewność).
    Epsilon 0.06 zapobiega klasyfikowaniu miękkich obiektów jako wielokąt.
    Pewność = solidity (stosunek area do hull area).
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "unknown", 0.0

    approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True)
    v = len(approx)
    area = cv2.contourArea(contour)

    if v == 3:
        shape = "triangle"
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.80 <= ar <= 1.25 else "rectangle"
    elif v == 5:
        shape = "pentagon"
    elif v == 6:
        shape = "hexagon"
    else:
        # Dużo wierzchołków — sprawdź cyrkularność
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        shape = "circle" if circularity > 0.72 else "polygon"

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    return shape, round(solidity, 2)


def _check_shape_filters(cnt) -> tuple[bool, dict]:
    """
    Sprawdza wszystkie filtry kształtu (area, AR, solidity, compactness, fill).
    Zwraca (czy_przechodzi, słownik_metryk).
    Wydzielone żeby uniknąć duplikacji między detekcją kolorową a detekcją Canny.
    """
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False, {}

    x, y, w, h = cv2.boundingRect(cnt)
    ar = max(w, h) / (min(w, h) + 1e-5)
    if ar > MAX_ASPECT_RATIO:
        return False, {}

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0.0
    if solidity < MIN_SOLIDITY:
        return False, {}

    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-5)
    if compactness > MAX_COMPACTNESS:
        return False, {}

    bbox_fill = area / (w * h + 1e-5)
    if bbox_fill < MIN_BBOX_FILL:
        return False, {}

    return True, {
        "area":         round(area, 1),
        "aspect_ratio": round(ar, 2),
        "compactness":  round(compactness, 2),
        "bbox_fill":    round(bbox_fill, 2),
    }


def compute_canny(frame: np.ndarray) -> np.ndarray:
    """
    Oblicza krawędzie Canny z adaptacyjnymi progami opartymi na medianie obrazu.
    Używane zarówno do detekcji ciemnych obiektów jak i trybu podglądu.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lo = int(max(10,  0.4 * median))
    hi = int(min(240, 1.3 * median))
    return cv2.Canny(blur, lo, hi)


def _detect_dark_shapes(frame: np.ndarray, hsv: np.ndarray,
                        skin: np.ndarray, existing: list[dict]) -> list[dict]:
    """
    Wykrywa ciemne obiekty (czarne słuchawki, granatowe/ciemnozielone przedmioty)
    metodą Canny + filtr jasności bezwzględnej.

    Obiekt zalicza się jako 'ciemny', gdy:
      - przechodzi filtry kształtu (solidity, compactness, etc.),
      - wnętrze konturu jest znacznie ciemniejsze niż średnia klatki,
      - nie pokrywa się ze zdetekowanym już obiektem kolorowym.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    lo = int(max(10,  0.4 * median))
    hi = int(min(240, 1.3 * median))
    edges = cv2.Canny(blur, lo, hi)

    # Dylatacja zamyka przerwy w krawędziach — łatwiej znaleźć zamknięte kontury
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v_channel = hsv[:, :, 2]
    global_mean_v = float(cv2.mean(v_channel)[0])
    # Próg jasności: obiekt musi być ciemniejszy niż X% średniej klatki
    dark_threshold = global_mean_v * DARK_BRIGHTNESS_RATIO

    results = []
    for cnt in contours:
        ok, metrics = _check_shape_filters(cnt)
        if not ok:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Pomiń jeśli środek blisko już wykrytego obiektu kolorowego
        if any(abs(cx - d["center"][0]) < 60 and abs(cy - d["center"][1]) < 60
               for d in existing):
            continue

        # Oblicz średnią jasność wewnątrz konturu
        obj_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, cv2.FILLED)
        mean_v = cv2.mean(v_channel, mask=obj_mask)[0]

        # Odrzuć jeśli wnętrze nie jest wyraźnie ciemne
        if mean_v > dark_threshold:
            continue

        # Odrzuć jeśli zdominowany przez kolor skóry (np. dłoń na ciemnym tle)
        skin_px = cv2.countNonZero(cv2.bitwise_and(skin, obj_mask))
        if skin_px / (metrics["area"] + 1e-5) > MAX_SKIN_RATIO:
            continue

        shape_name, confidence = classify_shape(cnt)

        results.append({
            "shape":        shape_name,
            "color":        "dark",
            "center":       (cx, cy),
            "area":         metrics["area"],
            "contour":      cnt,
            "confidence":   confidence,
            "aspect_ratio": metrics["aspect_ratio"],
            "compactness":  metrics["compactness"],
            "bbox_fill":    metrics["bbox_fill"],
        })

    return results


def detect_shapes(frame: np.ndarray) -> tuple[list[dict], dict]:
    """
    Detekcja dwuetapowa:
      1. Color-mask-first — kontury szukamy wewnątrz maski każdego koloru
         (tło jest ignorowane, kolory się nie mieszają).
      2. Canny + filtr jasności — dla ciemnych obiektów (czarnych, granatowych)
         które umykają detekcji kolorowej.

    Zwraca (detections, debug_info).
    debug_info: {nazwa_koloru: maska, "skin": maska, "canny": krawędzie}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = _build_skin_mask(hsv)
    debug_info: dict = {"skin": skin}
    results: list[dict] = []

    # Kernel 7x7 dla OPEN — agresywniej usuwa artefakty świetlne
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # Kernel 5x5 dla CLOSE — delikatniejsze wypełnianie luk
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for color_name, ranges in COLOR_RANGES.items():
        # Zbuduj maskę dla danego koloru ze wszystkich jego zakresów
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # OPEN usuwa małe artefakty świetlne, CLOSE wypełnia drobne dziury
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_kernel,  iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        debug_info[color_name] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            ok, metrics = _check_shape_filters(cnt)
            if not ok:
                continue

            # Filtr dłoni — odrzuć jeśli kontur zdominowany przez kolor skóry
            obj_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [cnt], -1, 255, cv2.FILLED)
            skin_px = cv2.countNonZero(cv2.bitwise_and(skin, obj_mask))
            if skin_px / (metrics["area"] + 1e-5) > MAX_SKIN_RATIO:
                continue

            shape_name, confidence = classify_shape(cnt)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            results.append({
                "shape":        shape_name,
                "color":        color_name,
                "center":       (cx, cy),
                "area":         metrics["area"],
                "contour":      cnt,
                "confidence":   confidence,
                "aspect_ratio": metrics["aspect_ratio"],
                "compactness":  metrics["compactness"],
                "bbox_fill":    metrics["bbox_fill"],
            })

    # Drugi etap: detekcja ciemnych obiektów (Canny + jasność)
    dark_results = _detect_dark_shapes(frame, hsv, skin, results)
    results.extend(dark_results)

    # Krawędzie Canny zapisane w debug_info dla trybu podglądu Canny
    debug_info["canny"] = compute_canny(frame)

    # Sortuj od największego do najmniejszego
    results.sort(key=lambda x: x["area"], reverse=True)
    return results, debug_info


def draw_detections(frame: np.ndarray, detections: list[dict],
                    debug: bool = False) -> np.ndarray:
    """
    Rysuje kontury, etykiety i punkty środkowe na klatce.
    W trybie debug wyświetla AR, compactness i fill pod każdą etykietą.
    """
    output = frame.copy()

    for det in detections:
        color_bgr = DRAW_COLORS.get(det["color"], DRAW_COLORS["unknown"])
        cx, cy = det["center"]

        # Kontur i punkt środkowy
        cv2.drawContours(output, [det["contour"]], -1, color_bgr, 2)
        cv2.circle(output, (cx, cy), 5, color_bgr, -1)

        # Etykieta: "shape (color)"
        label = f"{det['shape']} ({det['color']})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output,
                      (cx - tw // 2 - 4, cy - th - 10),
                      (cx + tw // 2 + 4, cy - 4),
                      color_bgr, -1)
        cv2.putText(output, label,
                    (cx - tw // 2, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Pewność pod etykietą
        conf_label = f"{int(det['confidence'] * 100)}%"
        cv2.putText(output, conf_label,
                    (cx - 10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        # Dodatkowe metryki w trybie debug
        if debug:
            extra = (f"ar={det['aspect_ratio']:.1f} "
                     f"cmp={det['compactness']:.1f} "
                     f"fill={det['bbox_fill']:.2f}")
            cv2.putText(output, extra,
                        (cx - tw // 2, cy + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 0), 1, cv2.LINE_AA)

    return output


def build_debug_overlay(frame: np.ndarray, debug_masks: dict) -> np.ndarray:
    """
    Nakłada kolorowe maski na klatkę (50% przezroczystości).
    Klucz 'canny' jest pomijany — obsługiwany osobno w trybie Canny.
    """
    overlay = np.zeros_like(frame)
    for name, mask in debug_masks.items():
        if name == "canny":
            continue  # Canny jest renderowany osobno
        bgr = _DEBUG_MASK_COLORS.get(name, (200, 200, 200))
        overlay[mask > 0] = bgr
    return cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)


def build_canny_view(frame: np.ndarray, canny: np.ndarray) -> np.ndarray:
    """
    Renderuje widok krawędzi Canny: biały linie na przyciemnionej klatce.
    Pozwala zobaczyć jednocześnie krawędzie i kontekst sceny.
    """
    # Przyciemnij oryginał do 30% — kontekst sceny bez dominowania
    dark_bg = (frame * 0.3).astype(np.uint8)
    # Nanieś białe linie tam gdzie Canny wykrył krawędź
    dark_bg[canny > 0] = (255, 255, 255)
    return dark_bg
