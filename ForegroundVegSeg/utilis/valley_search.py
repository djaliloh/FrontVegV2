import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def analyze_histogram_detect_valley(histogram, idx=None, label="Individual"):
    """
    Analyzes a histogram and detects the optimal valley between the peaks (modes).
    Author: Abdoul Djalil Ousseini Hamza

    Args:
        histogram (np.array): L'histogramme à analyser.
        idx (int, optional): L'indice de l'image, utilisé pour le log.
        label (str, optional): Nom descriptif pour affichage.

    Returns:
        tuple: (x, optimal_minimum_index) où x est l'axe des abscisses
               et optimal_minimum_index est l'index du minimum optimal détecté.
    """
    x = np.arange(len(histogram))
    y = histogram.copy()

    # Lissage optionnel
    smoothed = False
    if smoothed:
        y = gaussian_filter(y, sigma=2)

    # Peaks detection
    peaks, _ = find_peaks(y, height=0.5, prominence=0.5, distance=20)
    if len(peaks) < 3:
        print(f"Pas assez de pics détectés pour {label} (Image #{idx}).")
        return None, None

    # Detection of minima from the 2nd mode
    minima_indices = []
    for i in range(1, len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        min_index = np.argmin(y[start:end]) + start
        minima_indices.append(min_index)

    minima_indices = minima_indices[:4]  # Ne pas en garder plus de 4

    # Compute depths ("profondeur" de chaque minimum entre les modes (peaks)) 
    depths = []
    for i, min_index in enumerate(minima_indices):
        left_peak = peaks[i + 1]
        right_peak = peaks[i + 2] if i + 2 < len(peaks) else len(y) - 1
        depth = y[min_index] - np.mean([y[left_peak], y[right_peak]])
        depths.append((min_index, depth))

    # Select the optimal minimum : le plus profond
    optimal_minimum_index = None
    if depths:
        optimal_minimum_index = min(depths, key=lambda x: y[x[0]])[0]
        # print("Le minimum le plus profond est selectionné:", optimal_minimum_index)

    return x, optimal_minimum_index
