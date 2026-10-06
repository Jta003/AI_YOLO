# 🤖 AI_YOLO — YOLO26 Object Detection

โปรเจกต์ **Object Detection ด้วย YOLO26** สำหรับตรวจจับและจำแนกวัตถุจากภาพ วิดีโอ และกล้องแบบ Real-time โดยใช้โมเดล **YOLO26 Nano** และ **Ultralytics** เป็นหลัก

ระบบถูกออกแบบให้รองรับ Workflow ตั้งแต่การเตรียม Dataset จาก Annotation ไปจนถึงการ Train Model และนำ Model ที่ฝึกแล้วไปทดสอบกับภาพ วิดีโอ และกล้องแบบ Real-time

---

## 📌 Project Overview

โปรเจกต์นี้ใช้ **YOLO26** สำหรับตรวจจับวัตถุทั้งหมด **3 ประเภท**

| Class                  | Description          |
| ---------------------- | -------------------- |
| `miki_strawberry`      | Miki Strawberry      |
| `miki_strawberry_cola` | Miki Strawberry Cola |
| `milo`                 | Milo                 |

Model สามารถนำไปใช้ตรวจจับวัตถุจาก:

* 🖼️ Image
* 🎥 Video
* 📷 Webcam / Camera
* ⚡ Real-time Object Detection

---

## 🧠 Model

โปรเจกต์ใช้โมเดล:

```text
YOLO26 Nano
```

โดยเริ่มต้นจาก Pre-trained Model:

```text
yolo26n.pt
```

และนำมาฝึกต่อด้วย Dataset ของโปรเจกต์

---

## 🗂️ Project Structure

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
│   └── best.pt      
│
├── 01-export_dataset.py
├── 02-train.py
├── 03-test_image.py
├── 04-test_video.py
├── 05-test-camera.py
│
├── yolo26n.pt
└── .gitignore
```

---

## 🔄 Workflow

```text
Label Studio
      │
      ▼
Export Annotation JSON
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
YOLO26 Model
      │
      ├──────────────┐
      ▼              ▼
 Image Testing    Video Testing
      │              │
      └──────┬───────┘
             ▼
       Camera Testing
             │
             ▼
     Real-time Detection
```

---

## 🏷️ Dataset Preparation

ไฟล์ `01-export_dataset.py` ใช้สำหรับแปลง Annotation ที่ Export จาก **Label Studio** ให้เป็นรูปแบบ Dataset ที่ YOLO สามารถนำไปใช้ Train ได้

ระบบสามารถอ่าน Annotation แบบ Bounding Box และสร้างข้อมูลในรูปแบบ YOLO Detection

ตัวอย่าง Format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

ค่าพิกัดจะถูก Normalize ให้อยู่ในช่วง `0–1`

นอกจากนี้ Script ยังสามารถตรวจสอบ Class ที่มีอยู่ใน Annotation และแบ่ง Dataset เป็น Training และ Validation Set

ปัจจุบันกำหนดสัดส่วน Training:

```text
80%
```

และใช้ Seed:

```text
42
```

เพื่อให้การสุ่มสามารถทำซ้ำได้

---

## 🏋️ Training

ใช้ไฟล์:

```text
02-train.py
```

สำหรับ Train YOLO26 Model

Configuration หลัก:

```text
Model       : yolo26n.pt
Epochs      : 100
Image Size  : 640
Device      : GPU (device=0)
Optimizer   : MuSGD
```

### Data Augmentation

โปรเจกต์มีการใช้ Data Augmentation เพื่อช่วยให้ Model สามารถรับมือกับภาพที่มีมุมและลักษณะแตกต่างกันได้ดีขึ้น

```text
Rotation       : ±15°
Shear          : 5°
Perspective    : 0.001
Horizontal Flip: 50%
Vertical Flip  : Disabled
Mosaic         : Enabled
MixUp          : 0.1
```

และปิด Mosaic ในช่วงท้ายของการ Training:

```text
close_mosaic = 10
```

---

## 🖼️ Test with Image

ใช้:

```text
03-test_image.py
```

สำหรับทดสอบ Model กับภาพเดี่ยว

ตัวอย่างการใช้งาน:

```bash
python 03-test_image.py
```

ระบบจะทำการ Detection และแสดงผล Bounding Box บนภาพ

สามารถบันทึกผลลัพธ์ได้ด้วย:

```python
save=True
```

---

## 🎥 Test with Video

ใช้:

```text
04-test_video.py
```

สำหรับนำ Model ไปตรวจจับวัตถุจากไฟล์วิดีโอ

ตัวอย่าง:

```bash
python 04-test_video.py
```

ระบบสามารถ:

* อ่านวิดีโอ
* ตรวจจับวัตถุในแต่ละ Frame
* แสดงผลการ Detection
* บันทึกวิดีโอผลลัพธ์

ตัวอย่าง Confidence Threshold:

```text
conf = 0.5
```

ผลลัพธ์จะถูกบันทึกในโฟลเดอร์ของ Ultralytics เช่น:

```text
runs/detect/predict/
```

---

## 📷 Real-time Camera Detection

ใช้:

```text
05-test-camera.py
```

สำหรับตรวจจับวัตถุผ่านกล้องแบบ Real-time

เทคโนโลยีที่ใช้:

* OpenCV
* Ultralytics YOLO
* Python

การทำงาน:

```text
Camera
   │
   ▼
Capture Frame
   │
   ▼
YOLO26
   │
   ▼
Object Detection
   │
   ▼
Bounding Box + Class
   │
   ▼
Display
```

สามารถกด:

```text
Q
```

เพื่อออกจากโปรแกรม

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Jta003/AI_YOLO.git
cd AI_YOLO
```

### 2. Install Dependencies

ติดตั้ง Ultralytics:

```bash
pip install ultralytics
```

สำหรับการทำงานกับ Image Processing และ Dataset:

```bash
pip install pillow
```

และสำหรับ Real-time Camera:

```bash
pip install opencv-python
```

หรือสามารถติดตั้งทั้งหมด:

```bash
pip install ultralytics pillow opencv-python
```

---

## ▶️ Usage

### Train Model

```bash
python 02-train.py
```

### Test Image

```bash
python 03-test_image.py
```

### Test Video

```bash
python 04-test_video.py
```

### Test Camera

```bash
python 05-test-camera.py
```

---

## 🛠️ Technologies

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming Language      |
| YOLO26       | Object Detection          |
| Ultralytics  | YOLO Framework            |
| OpenCV       | Camera & Video Processing |
| Pillow       | Image Processing          |
| Label Studio | Dataset Annotation        |
| Git / GitHub | Version Control           |

---

## 📊 Detection Classes

```text
0 → miki_strawberry
1 → miki_strawberry_cola
2 → milo
```

ตัวอย่างผลลัพธ์:


---

## ✨ Features

* ✅ YOLO26 Object Detection
* ✅ Custom Dataset Training
* ✅ 3 Object Classes
* ✅ Label Studio Annotation Support
* ✅ Automatic Dataset Conversion
* ✅ Data Augmentation
* ✅ Image Detection
* ✅ Video Detection
* ✅ Real-time Camera Detection
* ✅ Bounding Box Visualization
* ✅ GPU Training Support

---

## 📁 Main Scripts

| File                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `01-export_dataset.py` | Convert Label Studio annotations to YOLO Dataset |
| `02-train.py`          | Train YOLO26 Model                               |
| `03-test_image.py`     | Test detection on images                         |
| `04-test_video.py`     | Test detection on videos                         |
| `05-test-camera.py`    | Real-time camera detection                       |
| `dataset/data.yaml`    | Dataset configuration                            |
| `yolo26n.pt`           | YOLO26 Nano pre-trained model                    |

---

## 🚀 Future Improvements

แนวทางที่สามารถพัฒนาต่อได้:

* [ ] เพิ่มจำนวน Dataset
* [ ] เพิ่มจำนวน Class
* [ ] ปรับปรุง Dataset ให้มีความหลากหลายมากขึ้น
* [ ] เพิ่ม Model Evaluation เช่น Precision, Recall และ mAP
* [ ] เพิ่มระบบ Object Tracking
* [ ] เพิ่ม Web / Application Interface
* [ ] Optimize Model สำหรับ Edge Device
* [ ] เพิ่ม Real-time FPS Monitoring
* [ ] เพิ่มระบบบันทึกผลการ Detection

---

## 👨‍💻 Author

**Jta003** **queyyz**

GitHub:

https://github.com/Jta003

---

## 📄 License

This project is intended for educational and development purposes.
