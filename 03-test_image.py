from ultralytics import YOLO

model = YOLO(
    r"D:\Codeจารรุจิ\AI_YOLO\best.pt"
)

results = model.predict(
    r"D:\Codeจารรุจิ\AI_YOLO\dataset\images\train\0012.jpg",
    conf=0.01,
    save=True
)

results[0].show()