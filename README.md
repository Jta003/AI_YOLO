# YOLO26 Object Detection

โปรเจกต์ **Object Detection ด้วย YOLO26** สำหรับตรวจจับวัตถุจากภาพและ Webcam แบบ Real-time โดยใช้ **Ultralytics YOLO** ร่วมกับ **OpenCV และ PyTorch**

โปรเจกต์นี้ครอบคลุมตั้งแต่การเตรียม Dataset จากการทำ Annotation, การแปลง Annotation ให้อยู่ในรูปแบบ YOLO, การ Train Model และการนำ Model ที่ Train แล้วไปทดสอบกับรูปภาพ วิดีโอ และ Webcam

---

## Project Overview

ระบบใช้ **YOLO26 Nano (`yolo26n.pt`)** เป็น Pre-trained Model และนำมาฝึกเพิ่มเติมด้วย Dataset ที่จัดเตรียมไว้สำหรับโปรเจกต์

Workflow หลักของระบบ:

```text
Annotation / Dataset
        │
        ▼
01-export_dataset.py
        │
        ▼
YOLO Dataset
        │
        ▼
02-train.py
        │
        ▼
Trained YOLO26 Model
        │
        ├───────────────┐
        ▼               ▼
03-test_image.py   04-test_video.py
        │               │
        │               ▼
        │          Video Detection
        │
        ▼
Image Detection

        └───────────────► 05-test-camera.py
                           │
                           ▼
                     Real-time Webcam
```

---

# Features

* Object Detection ด้วย YOLO26
* ใช้ Pre-trained Model `yolo26n.pt`
* รองรับ Dataset ที่สร้างจาก Annotation
* แปลง Annotation จาก Label Studio เป็น YOLO Format
* Train Model ด้วย Ultralytics
* Data Augmentation ระหว่าง Training
* ทดสอบ Model กับรูปภาพ
* ทดสอบ Model กับ Video
* ตรวจจับวัตถุจาก Webcam แบบ Real-time
* แสดง Bounding Box และ Class ของวัตถุที่ตรวจพบ

---

# Technology Stack

| Technology   | ใช้สำหรับ                          |
| ------------ | ---------------------------------- |
| Python       | พัฒนาโปรแกรม                       |
| YOLO26       | Object Detection                   |
| Ultralytics  | Training และ Inference             |
| PyTorch      | Deep Learning                      |
| OpenCV       | Image / Video / Webcam Processing  |
| Label Studio | Annotation และ Bounding Box        |
| Git / GitHub | Version Control และจัดเก็บโปรเจกต์ |

---

# Project Structure

โครงสร้างหลักของ Repository:

```text
AI_YOLO/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   │
│   └── data.yaml
│
├── models/
│
├── 01-export_dataset.py
├── 02-train.py
├── 03-test_image.py
├── 04-test_video.py
├── 05-test-camera.py
│
├── yolo26n.pt
│
└── .gitignore
```

Repository ปัจจุบันมีไฟล์หลักสำหรับ Export Dataset, Training, Image Testing, Video Testing และ Camera Testing อยู่ครบตาม workflow ของโปรเจกต์

---

# File Description

| File / Folder          | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `dataset/`             | Dataset สำหรับใช้ Train และ Validate Model         |
| `dataset/images/`      | รูปภาพสำหรับ Training และ Validation               |
| `dataset/labels/`      | Annotation ในรูปแบบ YOLO                           |
| `dataset/data.yaml`    | กำหนด Dataset Path และ Class                       |
| `models/`              | โฟลเดอร์สำหรับเก็บ Model                           |
| `01-export_dataset.py` | แปลง Annotation จาก Label Studio เป็น YOLO Dataset |
| `02-train.py`          | Train YOLO26 Model                                 |
| `03-test_image.py`     | ทดสอบ Model กับรูปภาพ                              |
| `04-test_video.py`     | ทดสอบ Model กับ Video                              |
| `05-test-camera.py`    | ตรวจจับวัตถุจาก Webcam แบบ Real-time               |
| `yolo26n.pt`           | Pre-trained YOLO26 Nano Model                      |
| `.gitignore`           | กำหนดไฟล์และโฟลเดอร์ที่ไม่ต้องการให้ Git ติดตาม    |

---

# Dataset

Dataset ใช้สำหรับฝึกให้ Model เรียนรู้ลักษณะของวัตถุที่ต้องการตรวจจับ

โครงสร้าง Dataset:

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

ไฟล์ Label ใช้รูปแบบ YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

ค่าพิกัดทั้งหมดเป็นค่า Normalized อยู่ในช่วง `0 - 1`

---

# Dataset Preparation

ไฟล์:

```text
01-export_dataset.py
```

ใช้สำหรับนำข้อมูล Annotation ที่ Export จาก **Label Studio** มาแปลงเป็น YOLO Dataset

โปรแกรมจะ:

1. อ่านไฟล์ `.json`
2. ตรวจสอบ Annotation
3. ค้นหา Class ที่ใช้ใน Dataset
4. แปลง Bounding Box เป็น YOLO Format
5. แบ่ง Dataset เป็น `train` และ `val`
6. สร้างโครงสร้าง Dataset สำหรับใช้กับ YOLO

ในโปรเจกต์กำหนดให้ใช้:

```python
TRAIN_SPLIT = 0.8
SEED = 42
```

หมายความว่า Dataset ถูกแบ่งประมาณ:

```text
80% → Training
20% → Validation
```

Script ยังรองรับทั้ง YOLO Detection และ YOLO-OBB โดยค่าปัจจุบันตั้งเป็น:

```python
USE_OBB = False
```

ดังนั้นระบบจะใช้ Bounding Box แบบปกติของ YOLO Detection

---

# Training

ไฟล์:

```text
02-train.py
```

ใช้สำหรับ Train Model โดยเริ่มจาก Pre-trained:

```text
yolo26n.pt
```

การตั้งค่าหลักของ Training:

```text
Model       : YOLO26 Nano
Epochs      : 100
Image Size  : 640
Optimizer   : MuSGD
Device      : GPU 0
```

การ Train ในโปรเจกต์มีการใช้ Data Augmentation เพื่อช่วยให้ Model สามารถรับมือกับภาพที่มีมุมและสภาพแตกต่างกันได้ดีขึ้น

ตัวอย่าง Augmentation ที่ใช้:

```text
Rotation       : ±15°
Shear          : 5°
Perspective    : 0.001
Horizontal Flip: 50%
Vertical Flip  : 0%
Mosaic         : 1.0
MixUp          : 0.1
Close Mosaic   : 10 epochs
```

ค่าดังกล่าวถูกกำหนดไว้ใน `02-train.py` ของ Repository ปัจจุบัน

---

# Train Model

หลังจากเตรียม Dataset แล้ว สามารถเริ่ม Training ได้ด้วย:

```bash
python 02-train.py
```

หรือใช้ Ultralytics CLI:

```bash
yolo detect train data="dataset/data.yaml" model="yolo26n.pt" epochs=100 imgsz=640
```

> หมายเหตุ: Path ของ Dataset ใน `02-train.py` อาจต้องเปลี่ยนให้ตรงกับตำแหน่งโปรเจกต์บนเครื่องของผู้ใช้งาน

---

# Image Detection

ไฟล์:

```text
03-test_image.py
```

ใช้สำหรับทดสอบ Model กับรูปภาพ

Workflow:

```text
Input Image
     ↓
YOLO26 Model
     ↓
Object Detection
     ↓
Bounding Box
     ↓
Detection Result
```

ตัวอย่างการใช้งาน:

```bash
python 03-test_image.py
```

ภายใน Script จะโหลด Model แล้วใช้ `model.predict()` เพื่อทำการตรวจจับวัตถุจากรูปภาพ พร้อมแสดงผลลัพธ์ Bounding Box

---

# Video Detection

ไฟล์:

```text
04-test_video.py
```

ใช้สำหรับทดสอบ Model กับไฟล์ Video

Workflow:

```text
Video
  ↓
OpenCV
  ↓
YOLO26
  ↓
Object Detection
  ↓
Bounding Box
  ↓
Output Video
```

รันด้วย:

```bash
python 04-test_video.py
```

---

# Real-time Webcam Detection

ไฟล์:

```text
05-test-camera.py
```

ใช้สำหรับตรวจจับวัตถุจาก Webcam แบบ Real-time

Workflow:

```text
Webcam
   ↓
OpenCV
   ↓
YOLO26
   ↓
Object Detection
   ↓
Bounding Box + Class
   ↓
Display
```

รัน:

```bash
python 05-test-camera.py
```

โปรแกรมจะเปิด Webcam และประมวลผลภาพแต่ละ Frame ด้วย YOLO Model

กด:

```text
q
```

เพื่อออกจากโปรแกรม

ใน Script ปัจจุบันใช้:

```python
cv2.VideoCapture(1)
```

และใช้ CPU สำหรับ Inference:

```python
device='cpu'
```

ดังนั้นหากต้องการใช้กล้องตัวอื่น สามารถเปลี่ยนหมายเลข Camera ได้ เช่น:

```python
cv2.VideoCapture(0)
```

หรือ:

```python
cv2.VideoCapture(1)
```

---

# Model

โปรเจกต์ใช้:

```text
yolo26n.pt
```

เป็น Pre-trained Model สำหรับเริ่มต้น Training

หลังจาก Train เสร็จ Model ที่ได้สามารถนำไปใช้สำหรับ:

* Image Detection
* Video Detection
* Real-time Webcam Detection

ตัวอย่าง Path ของ Trained Model:

```text
runs/
└── detect/
    └── ice-cream_obb/
        └── weights/
            └── best.pt
```

> ชื่อโฟลเดอร์ของ Training Run อาจเปลี่ยนไปตามการตั้งค่าและการ Train แต่ละครั้ง

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Jta003/AI_YOLO.git
cd AI_YOLO
```

## 2. Create Virtual Environment

```bash
python -m venv env
```

### Windows PowerShell

```powershell
.\env\Scripts\Activate.ps1
```

## 3. Install Dependencies

ติดตั้ง Package ที่จำเป็น:

```bash
pip install ultralytics opencv-python torch torchvision pillow
```

---

# Recommended Workflow

หากต้องการเริ่มต้นโปรเจกต์ตั้งแต่ Dataset:

```text
1. Collect Images
       ↓
2. Annotation
       ↓
3. Export Dataset
       ↓
4. Train YOLO26
       ↓
5. Validate Model
       ↓
6. Test Image
       ↓
7. Test Video
       ↓
8. Test Webcam
```

สามารถรันตามลำดับ:

```bash
python 01-export_dataset.py
```

จากนั้น:

```bash
python 02-train.py
```

ทดสอบรูปภาพ:

```bash
python 03-test_image.py
```

ทดสอบ Video:

```bash
python 04-test_video.py
```

ทดสอบ Webcam:

```bash
python 05-test-camera.py
```

---

# Model Evaluation

หลังจาก Training สามารถตรวจสอบประสิทธิภาพของ Model ด้วย Validation:

```bash
yolo detect val model="runs/detect/ice-cream_obb/weights/best.pt" data="dataset/data.yaml" imgsz=640
```

Metrics ที่สามารถใช้ประเมิน Model ได้แก่:

| Metric    | Description                          |
| --------- | ------------------------------------ |
| Precision | ความแม่นยำของการตรวจจับ              |
| Recall    | ความสามารถในการตรวจจับวัตถุที่มีอยู่ |
| mAP50     | Mean Average Precision ที่ IoU 0.50  |
| mAP50-95  | ค่า mAP ที่ประเมินหลายระดับ IoU      |

---

# Data Augmentation

โปรเจกต์มีการใช้ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล Training

ตัวอย่าง:

```text
Original Image
      │
      ├── Rotation
      ├── Shear
      ├── Perspective
      ├── Horizontal Flip
      ├── Mosaic
      └── MixUp
             │
             ▼
      Augmented Images
             │
             ▼
        YOLO Training
```

การเพิ่มความหลากหลายของภาพช่วยลดปัญหา Model เรียนรู้จากภาพ Training มากเกินไป และช่วยให้ Model สามารถรับมือกับภาพที่มีมุมมองแตกต่างจาก Dataset ได้ดีขึ้น

---

# Tips for Better Detection

เพื่อเพิ่มประสิทธิภาพของ Model ควรมี Dataset ที่หลากหลาย เช่น:

* มุมกล้องหลายมุม
* ระยะใกล้และระยะไกล
* พื้นหลังหลายรูปแบบ
* สภาพแสงแตกต่างกัน
* วัตถุที่มีขนาดแตกต่างกัน
* วัตถุหลายชิ้นในภาพเดียว
* ภาพจาก Webcam จริง
* ภาพที่มีการบังบางส่วนของวัตถุ

ควรแยก Dataset สำหรับ:

```text
Training
Validation
Testing
```

เพื่อให้สามารถประเมิน Model กับข้อมูลที่ Model ไม่เคยเห็นมาก่อนได้อย่างเหมาะสม

---

# Project Workflow

```text
                 ┌────────────────────┐
                 │   Image Dataset    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Label Studio       │
                 │ Annotation         │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ 01-export_dataset  │
                 │       .py          │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │    YOLO Dataset    │
                 │ train / val        │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │    02-train.py     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   Trained Model    │
                 │      best.pt       │
                 └─────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────────┐
       │  Image   │  │  Video   │  │   Webcam     │
       │ Testing  │  │ Testing  │  │ Real-time    │
       └────┬─────┘  └────┬─────┘  └──────┬───────┘
            │             │               │
            ▼             ▼               ▼
       03-test_      04-test_        05-test-
       image.py      video.py        camera.py
```

---

# Project Status

**Status: Completed / Ready for Testing**

โปรเจกต์มีระบบหลักสำหรับ:

* Dataset Preparation
* Annotation Conversion
* YOLO26 Training
* Image Detection
* Video Detection
* Real-time Webcam Detection

พร้อมสำหรับการนำ Model ที่ Train แล้วไปทดสอบและพัฒนาต่อ

---

# Notes

* Path ที่อยู่ใน Python Script เป็น Path ของเครื่องผู้พัฒนา ดังนั้นอาจต้องแก้ไขเมื่อ Clone Repository ไปใช้บนเครื่องอื่น
* `yolo26n.pt` เป็น Pre-trained Model ที่ใช้สำหรับเริ่มต้น Training
* `best.pt` คือ Model ที่ได้หลังจาก Training และควรใช้ Model ที่มีผล Validation ดีที่สุดสำหรับ Inference
* หาก Webcam ทำงานไม่ถูกต้อง ให้ลองเปลี่ยน Camera Index จาก `1` เป็น `0`
* หากต้องการความเร็วในการตรวจจับมากขึ้น สามารถพิจารณาใช้ GPU แทน CPU ในขั้นตอน Inference
* Dataset ที่มีความหลากหลายจะช่วยให้ Model ทำงานได้ดีขึ้นในสภาพแวดล้อมจริง

---

# Repository

GitHub Repository:

https://github.com/Jta003/AI_YOLO

---

# Summary

โปรเจกต์นี้เป็นระบบ **Object Detection ด้วย YOLO26** ที่ครอบคลุมตั้งแต่การเตรียม Dataset ไปจนถึงการนำ Model ไปใช้งานจริง

```text
Dataset
   ↓
Annotation
   ↓
YOLO Format
   ↓
YOLO26 Training
   ↓
best.pt
   ↓
┌───────────────┬───────────────┬────────────────┐
│ Image         │ Video         │ Webcam         │
│ Detection     │ Detection     │ Real-time      │
└───────────────┴───────────────┴────────────────┘
```

โปรเจกต์ถูกพัฒนาเพื่อเป็นระบบ Object Detection ที่สามารถนำไปต่อยอดกับงาน Computer Vision และ Real-time Object Detection ได้
