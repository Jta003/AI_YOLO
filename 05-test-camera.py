import cv2
from ultralytics import YOLO


def main():
    # โหลดโมเดลที่ผ่านการฝึก (Trained Model)
    model = YOLO(r"D:\Codeจารรุจิ\AI_YOLO\runs\detect\ice-cream_obb\weights\best.pt")

    # เปิดใช้งานกล้องเว็บแคม
    # เลข 0 หมายถึงกล้องตัวแรกที่เชื่อมต่อกับคอมพิวเตอร์
    cap = cv2.VideoCapture(1)

    # ตรวจสอบว่าสามารถเปิดกล้องได้หรือไม่
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    # แสดงคำแนะนำการใช้งาน
    print("กด 'q' เพื่อออกจากโปรแกรม")

    # เริ่มการตรวจจับแบบ Real-time
    while True:
        # อ่านภาพจากกล้องทีละเฟรม
        success, frame = cap.read()

        # ตรวจสอบว่าสามารถอ่านภาพจากกล้องได้
        if success:

            # นำภาพจากกล้องเข้าสู่โมเดล YOLO
            results = model.predict(
                source=frame,
                stream=True,    # ประมวลผลแบบต่อเนื่อง เหมาะสำหรับวิดีโอหรือ Real-time
                device='cpu'    # ใช้ CPU ในการประมวลผล
            )

            # วนลูปเพื่อรับผลลัพธ์จากโมเดล
            for r in results:
                # วาดกรอบ Bounding Box และชื่อ Class ลงบนภาพ
                annotated_frame = r.plot()

            # แสดงภาพที่ผ่านการตรวจจับบนหน้าต่าง
            cv2.imshow("YOLO26 Real-time Detection", annotated_frame)

            # กดปุ่ม 'q' เพื่อออกจากโปรแกรม
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # หากไม่สามารถอ่านภาพจากกล้องได้ ให้หยุดการทำงาน
        else:
            break

    # ปิดการเชื่อมต่อกับกล้อง
    cap.release()

    # ปิดหน้าต่างแสดงผลทั้งหมด
    cv2.destroyAllWindows()


# เรียกใช้งานฟังก์ชัน main()
if __name__ == '__main__':
    main()