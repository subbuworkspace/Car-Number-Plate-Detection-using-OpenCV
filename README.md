# 🚗 Car Number Plate Detection using OpenCV

A beginner-friendly **Computer Vision** project using **Python, OpenCV, and Haar Cascade Classifier** to detect vehicle number plates from images.

The project identifies the approximate location of a vehicle's number plate and draws a bounding box around the detected plate.

This project focuses on **number plate detection**, not Optical Character Recognition (OCR).

---

# 📌 Project Overview

The objective is to detect a car's number plate from an input image using a pre-trained Haar Cascade classifier.

### Detection Pipeline

```text
Input Car Image
       ↓
Read Image
       ↓
Convert to Grayscale
       ↓
Haar Cascade Classifier
       ↓
Detect Number Plate
       ↓
Draw Bounding Box
       ↓
Crop Number Plate
       ↓
Display / Save Result
```

---

# 🧠 What is Number Plate Detection?

Number Plate Detection is a Computer Vision task where the system identifies the location of a vehicle's registration plate in an image or video.

For example:

```text
          CAR
┌───────────────────────────┐
│                           │
│        🚗                 │
│                           │
│       ┌─────────────┐     │
│       │ WB 12 AB    │     │
│       │   1234      │     │
│       └─────────────┘     │
│        Number Plate       │
└───────────────────────────┘
```

The detector identifies the rectangular number-plate region.

---

# 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Haar Cascade Classifier
* Computer Vision
* Git
* GitHub

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Car-Number-Plate-Detection-OpenCV.git
```

Move into the project:

```bash
cd Car-Number-Plate-Detection-OpenCV
```

---

# 🐍 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Command Prompt:

```cmd
.venv\Scripts\activate
```

---

# 📥 3. Install Dependencies

```bash
python -m pip install opencv-python==4.10.0.84
python -m pip install numpy
```

Or:

```bash
python -m pip install -r requirements.txt
```

---

# 📁 Project Structure

```text
Car-Number-Plate-Detection-OpenCV/
│
├── Haarcascades/
│   └── haarcascade_russian_plate_number.xml
│
├── images/
│   └── car.jpg
│
├── output/
│   └── detected_plate.jpg
│
├── number_plate_detection.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

> **Note:** A number-plate Haar Cascade XML file must be included in the `Haarcascades` folder, and the Python code must point to the correct XML file.

---

# 🔍 Haar Cascade for Number Plate Detection

A Haar Cascade classifier can be trained to detect specific objects.

For number plates, the classifier looks for visual patterns associated with plate-like regions.

Example:

```text
Car Image
    ↓
Sliding Detection Window
    ↓
Haar Features
    ↓
Classifier
    ↓
Plate Detected
```

---

# 💻 Basic Python Implementation

```python
import cv2

# ---------------------------------------
# 1. Load Haar Cascade
# ---------------------------------------

plate_path = r"Haarcascades\haarcascade_russian_plate_number.xml"

plate_classifier = cv2.CascadeClassifier(plate_path)

# Check classifier
if plate_classifier.empty():
    print("Error: Number plate Haar Cascade not loaded!")
    exit()

print("Number plate Haar Cascade loaded successfully!")


# ---------------------------------------
# 2. Load Car Image
# ---------------------------------------

image_path = r"images\car.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

print("Image loaded successfully!")


# ---------------------------------------
# 3. Convert to Grayscale
# ---------------------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------
# 4. Detect Number Plate
# ---------------------------------------

plates = plate_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4
)


# ---------------------------------------
# 5. Check Detection
# ---------------------------------------

if len(plates) == 0:
    print("No number plate detected!")

else:
    print("Number plate(s) detected:", len(plates))


# ---------------------------------------
# 6. Draw Rectangle and Crop Plate
# ---------------------------------------

for i, (x, y, w, h) in enumerate(plates):

    # Draw rectangle
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    # Crop plate
    plate = img[y:y+h, x:x+w]

    # Save cropped plate
    output_path = f"output/plate_{i + 1}.jpg"

    cv2.imwrite(output_path, plate)

    print("Plate saved:", output_path)


# ---------------------------------------
# 7. Display Result
# ---------------------------------------

cv2.imshow("Car Number Plate Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# 🔎 Understanding the Code

## Load Haar Cascade

```python
plate_classifier = cv2.CascadeClassifier(plate_path)
```

This loads the trained number-plate detector.

---

## Read Image

```python
img = cv2.imread(image_path)
```

OpenCV reads the car image into a NumPy array.

---

## Convert to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Haar Cascade detection works with grayscale intensity information.

---

## Detect Plate

```python
plates = plate_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4
)
```

This searches the image at multiple scales.

---

# ⚙️ Important Parameters

## `scaleFactor`

Example:

```python
scaleFactor=1.1
```

This controls how much the image is resized during the multi-scale search.

Smaller values:

```text
1.05
1.1
```

Can detect objects at more scales but require more processing.

Larger values:

```text
1.3
1.5
```

Can be faster but may miss objects.

---

# `minNeighbors`

Example:

```python
minNeighbors=4
```

This controls how many nearby detections are required before accepting a detection.

Lower value:

```python
minNeighbors=3
```

May detect more plates but can produce false positives.

Higher value:

```python
minNeighbors=6
```

Can reduce false positives but may miss plates.

---

# 🎯 Region of Interest

After detecting the plate:

```python
plate = img[y:y+h, x:x+w]
```

we crop only the detected plate.

Example:

```text
Original Image

┌─────────────────────────────┐
│                             │
│          🚗                 │
│                             │
│       ┌─────────────┐       │
│       │ WB12AB1234  │       │
│       └─────────────┘       │
│                             │
└─────────────────────────────┘
             ↓
          Crop ROI
             ↓
┌─────────────────────┐
│ WB12AB1234          │
└─────────────────────┘
```

---

# 💾 Saving the Plate

```python
cv2.imwrite(output_path, plate)
```

This saves the detected plate as an image.

Example:

```text
output/
│
├── plate_1.jpg
├── plate_2.jpg
└── plate_3.jpg
```

---

# 🧠 Number Plate Detection vs Number Plate Recognition

These are two different tasks.

## 1. Number Plate Detection

Question:

> Where is the number plate?

Output:

```text
Bounding Box
```

Example:

```text
┌────────────────────┐
│ WB 12 AB 1234      │
└────────────────────┘
```

---

## 2. Number Plate Recognition / OCR

Question:

> What characters are written on the plate?

Output:

```text
WB12AB1234
```

OCR technologies include:

* Tesseract
* EasyOCR
* PaddleOCR
* Deep Learning OCR

---

# 🔥 Complete ANPR Pipeline

A real-world **Automatic Number Plate Recognition (ANPR)** system can look like:

```text
Camera
  ↓
Vehicle Detection
  ↓
Number Plate Detection
  ↓
Plate Cropping
  ↓
Image Preprocessing
  ↓
OCR
  ↓
Text Cleaning
  ↓
Number Plate
```

Example:

```text
Car
 ↓
Number Plate
 ↓
Crop
 ↓
Grayscale
 ↓
Threshold
 ↓
Noise Removal
 ↓
OCR
 ↓
"WB12AB1234"
```

---

# 🔬 Image Preprocessing for OCR

After detecting the plate, we can improve it before OCR.

```python
plate_gray = cv2.cvtColor(
    plate,
    cv2.COLOR_BGR2GRAY
)
```

Apply thresholding:

```python
_, thresh = cv2.threshold(
    plate_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

Then:

```python
cv2.imshow("Plate", thresh)
```

---

# 🔤 OCR Example Using Tesseract

Install:

```bash
pip install pytesseract
```

Then:

```python
import pytesseract

text = pytesseract.image_to_string(
    plate_gray,
    config="--psm 7"
)

print("Detected Number Plate:", text)
```

A complete ANPR pipeline can therefore be:

```text
OpenCV
   ↓
Haar Cascade
   ↓
Number Plate Detection
   ↓
Crop
   ↓
Grayscale
   ↓
Threshold
   ↓
Tesseract OCR
   ↓
Plate Text
```

---

# 🆚 Haar Cascade vs Modern Number Plate Detection

| Feature            | Haar Cascade        | YOLO / Deep Learning        |
| ------------------ | ------------------- | --------------------------- |
| Approach           | Traditional ML      | Deep Learning               |
| Training           | Pre-trained cascade | Neural network              |
| Speed              | Fast                | Fast with suitable hardware |
| Accuracy           | Moderate            | Generally higher            |
| Difficult angles   | Limited             | Better                      |
| Lighting variation | Limited             | More robust                 |
| Small plates       | Can struggle        | Generally better            |
| Beginner friendly  | Very easy           | Moderate                    |
| CPU requirements   | Low                 | Higher                      |
| Real-world ANPR    | Limited             | Common approach             |

---

# ⚠️ Limitations

Haar Cascade number-plate detection can have problems with:

* Different plate designs
* Different plate sizes
* Poor lighting
* Blurred images
* Dirty plates
* Tilted plates
* Partially hidden plates
* Multiple vehicles
* Different camera angles
* Low-resolution images

Therefore, this project is primarily useful for **learning Computer Vision fundamentals**.

For production ANPR systems, modern object-detection and OCR models are generally more suitable.

---

# 🚘 Multiple Vehicle Detection

The same approach can detect multiple plates.

```python
for (x, y, w, h) in plates:

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )
```

Output:

```text
Car 1 → Plate detected
Car 2 → Plate detected
Car 3 → Plate detected
```

---

# 🎥 Future Improvement: Real-Time Webcam

Instead of an image:

```python
img = cv2.imread("car.jpg")
```

use:

```python
cap = cv2.VideoCapture(0)
```

Then continuously read frames:

```python
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    plates = plate_classifier.detectMultiScale(
        gray,
        1.1,
        4
    )

    for (x, y, w, h) in plates:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Number Plate Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Press:

```text
Q
```

to stop the webcam.

---

# 🌐 Future Improvement: Streamlit

The project can be converted into a web application:

```text
        Streamlit
            ↓
     Upload Car Image
            ↓
       OpenCV
            ↓
   Haar Cascade Detector
            ↓
    Number Plate Detection
            ↓
       Display Result
```

Possible interface:

```text
┌─────────────────────────────────┐
│     🚗 Number Plate Detector    │
├─────────────────────────────────┤
│                                 │
│     [ Upload Car Image ]        │
│                                 │
│          ↓                      │
│                                 │
│      Detected Plate             │
│      ┌──────────────┐           │
│      │ WB12AB1234   │           │
│      └──────────────┘           │
│                                 │
└─────────────────────────────────┘
```

---

# 🤗 Future Improvement: Gradio

The same detection model can be exposed through a simple Gradio interface:

```text
Image Upload
     ↓
Gradio
     ↓
OpenCV
     ↓
Haar Cascade
     ↓
Detected Number Plate
     ↓
Output Image
```

---

# 📊 Computer Vision Concepts Covered

This project covers:

* Image loading
* Image representation
* BGR color space
* Grayscale conversion
* Haar Cascade
* Object detection
* Multi-scale detection
* Bounding boxes
* Region of Interest
* Image cropping
* Image preprocessing
* Thresholding
* OCR concepts
* Real-time detection
* OpenCV

---

# 🎤 Interview Questions

## Beginner

### 1. What is number plate detection?

It is the process of locating a vehicle's registration plate in an image or video.

### 2. What is ANPR?

ANPR stands for **Automatic Number Plate Recognition**.

It combines number-plate detection and character recognition.

### 3. What is OpenCV?

OpenCV is an open-source Computer Vision library.

### 4. What is Haar Cascade?

Haar Cascade is a traditional object-detection technique based on Haar-like features and a cascade of classifiers.

### 5. What is a bounding box?

A rectangular region indicating the location of a detected object.

### 6. What is ROI?

ROI stands for **Region of Interest**.

It is the part of an image selected for further processing.

---

# 🎤 Intermediate Interview Questions

### 7. Why convert the image to grayscale?

It reduces the image to one intensity channel and makes traditional detection computationally simpler.

### 8. What is `detectMultiScale()`?

It detects objects at different sizes within an image.

### 9. What does `scaleFactor` do?

It controls the reduction between successive image scales used during detection.

### 10. What does `minNeighbors` do?

It determines how many neighboring detections are needed before accepting a detection.

### 11. What is a false positive?

The system detects a plate where there is no actual plate.

### 12. What is a false negative?

The system fails to detect an actual plate.

---

# 🎤 Advanced Interview Questions

### 13. What is the difference between detection and recognition?

**Detection:**

```text
Where is the plate?
```

**Recognition:**

```text
What characters are written on the plate?
```

### 14. What is OCR?

OCR stands for **Optical Character Recognition**. It converts characters in an image into machine-readable text.

### 15. Why is preprocessing important before OCR?

Noise, blur, poor contrast and uneven lighting can make character recognition difficult. Preprocessing can improve the quality of the text region.

### 16. Why might Haar Cascade fail for number plates?

Because it is sensitive to variations in:

* Scale
* Rotation
* Lighting
* Image quality
* Plate design
* Camera angle

### 17. How can you improve this project?

Possible improvements include:

```text
Haar Cascade
      ↓
YOLO
      ↓
Plate Detection
      ↓
Image Preprocessing
      ↓
EasyOCR / Tesseract
      ↓
Text Validation
      ↓
ANPR System
```

---

# 📄 requirements.txt

```text
opencv-python==4.10.0.84
numpy
```

For OCR:

```text
pytesseract
```

---

# 🔐 .gitignore

```text
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

# ▶️ Run the Project

Activate the environment:

```bash
source .venv/Scripts/activate
```

Run:

```bash
python number_plate_detection.py
```

---

# 📌 Learning Roadmap

```text
OpenCV
   ↓
Haar Cascade
   ↓
Face Detection
   ↓
Number Plate Detection
   ↓
Plate Cropping
   ↓
Image Preprocessing
   ↓
OCR
   ↓
ANPR
   ↓
YOLO
   ↓
Real-Time ANPR
   ↓
Streamlit / Gradio
```

---

# ⭐ Project Extensions

You can extend this project into:

* 🚗 Vehicle Detection
* 🔢 Number Plate Detection
* ✂️ Number Plate Cropping
* 🔤 OCR
* 📹 Real-Time ANPR
* 🗃️ Vehicle Database
* 🕐 Entry/Exit Time Tracking
* 🚦 Parking Management
* 🏢 Automatic Gate System
* 📊 Streamlit Dashboard
* 🤗 Gradio Application
* 🎯 YOLO-based Number Plate Detection

---

# 👨‍💻 Author

**Subrata Mondal**

Data Analyst | Data Science | Machine Learning | Artificial Intelligence | Computer Vision

---

# ⭐ Key Takeaway

The core concept of this project is:

```text
Car Image
    ↓
OpenCV
    ↓
Grayscale
    ↓
Haar Cascade
    ↓
Number Plate Detection
    ↓
Bounding Box
    ↓
Crop Plate
    ↓
OCR
    ↓
Number Plate Text
```

This project provides a foundation for building a more advanced **Automatic Number Plate Recognition (ANPR)** system using modern Computer Vision and Deep Learning techniques.
