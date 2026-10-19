# Dataset Directory Layout & YOLO Annotation Format

## 1. Directory Structure

YOLO expects images and their corresponding label files to share the **exact same filename** (only the extension changes, e.g., `frame_001.jpg` $\leftrightarrow$ `frame_001.txt`).

```text
data/
├── data.yaml            <-- Config file telling YOLO where to find images
├── train/
│   ├── images/
│   │   ├── flame_01.jpg
│   │   └── normal_01.jpg
│   └── labels/
│       ├── flame_01.txt
│       └── normal_01.txt  <-- Empty file if normal_01 has NO fire!
├── val/
│   ├── images/
│   └── labels/
└── test/               <-- Optional
    ├── images/
    └── labels/
```

---

## 2. Label File Format (`.txt`)

Each line in a `.txt` file represents **one bounding box** around an object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Example: `flame_01.txt`
```text
0 0.452 0.612 0.180 0.250
1 0.470 0.420 0.310 0.350
```
- Line 1: Class `0` (`fire`), centered at 45.2% width, 61.2% height, taking 18.0% of image width and 25.0% of image height.
- Line 2: Class `1` (`smoke`), centered at 47.0% width, 42.0% height.

---

## 3. Critical Rules for YOLO Datasets

1. **Normalized Coordinates (0.0 to 1.0):**
   - Coordinates are NOT raw pixel values (like 640 or 1080).
   - $x_{\text{center}} = \frac{x_{\text{pixel}}}{W_{\text{image}}}$
   - $y_{\text{center}} = \frac{y_{\text{pixel}}}{H_{\text{image}}}$
   - $\text{width} = \frac{\text{box\_width}_{\text{pixel}}}{W_{\text{image}}}$
   - $\text{height} = \frac{\text{box\_height}_{\text{pixel}}}{H_{\text{image}}}$
2. **Negative Samples (Background Images):**
   - Always include images of boiler rooms with **no fire or smoke**!
   - For these images, create an **empty** `.txt` label file (0 bytes).
   - *Why?* If every training image contains fire, the model develops a bias and hallucinates fire in normal rooms.

