# 🧠 Image Classification and Identification – CLI-Based ML System

Welcome to our CSE299 Junior Design Project from North South University! This project provides a **unified command-line interface (CLI)** to run multiple real-world **image-based machine learning tasks** using Python and open-source libraries.

---

## 👨‍💻 Group Members

| Name               | ID           |
|--------------------|--------------|
| Noshin Nawar       | 2221507042   |
| Md. Arebi Sarker   | 2221362042   |
| Mubashshira Kaisar | 2221070642   |
| Md Zidan Khan      | 2231413642   |

---

## 📌 Project Objective

To design a Python-based command-line system that can **classify, detect, segment, and analyze** images through multiple ML tasks – all using one easy-to-use CLI.

---

## ⚙️ Supported Tasks

| Task Command       | Description |
|--------------------|-------------|
| `classify`         | Predict object class (e.g., dog/cat, traffic sign) |
| `segment`          | Label each pixel to identify regions/objects |
| `detect`           | Detect and locate objects using bounding boxes |
| `size`             | Estimate the real-world size of an object |
| `digit`            | Recognize handwritten digits and characters |
| `scan`             | Convert photos of documents to scanned format |
| `omr`              | Grade OMR sheets by detecting filled bubbles |
| `track`            | Track a ball’s motion across video frames |
| `drowsy`           | Detect driver fatigue using eye aspect ratio |
| `fracture`         | Identify bone fractures in medical X-ray images |

---

## 🚀 How to Run

1. **Install Python (if not already installed):**
   https://www.python.org/downloads/

2. **Open PowerShell or Command Prompt**  
   Navigate to the project folder:

   ```bash
   cd path\to\Junior_Design_Project

3. **Run a command:**  
   
   ```bash
   python main.py --task classify --input test.jpg

# 📁 Folder Structure
```bash
Junior_Design_Project/
│
├── main.py                # CLI entry point
├── README.md              # Project documentation
├── tasks/                 # Folder for individual task modules
│   ├── __init__.py
│   ├── classifier.py
│   ├── segmenter.py
│   ├── detector.py
│   ├── size_estimator.py
│   ├── digit_recognizer.py
│   ├── scanner.py
│   ├── omr_grader.py
│   ├── ball_tracker.py
│   ├── drowsiness.py
│   └── fracture_detector.py




