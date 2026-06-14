# test.py
# 가장 성능이 좋은 모델(test2 - YOLOv8s)로 실제 탐지 테스트
# test2017 이미지 6장에서 손상 위치와 종류를 탐지
from ultralytics import YOLO
import glob

base = '/home/ubuntu/CarDD_release/CarDD_COCO'

# best model
model = YOLO("runs/detect/test2/weights/best.pt")

# run detection on test images
test_images = glob.glob(f'{base}/test2017/*.jpg')[:6]
results = model(test_images)

# results = model(
#     source=f'{base}/test2017/',
#     conf=0.25,
#     save=True,
#     device=0
# )

# process results
for i, result in enumerate(results):
    filename = f"result_{i+1}.jpg"
    result.save(filename=filename)
    print(f"Saved: {filename}")
