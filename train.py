# train.py
# YOLOv8 모델 학습( 4가지 다른 설정으로 성능 비교 )
# Test 1: YOLOv8n baseline
# Test 2: YOLOv8s (모델 변경)
# Test 3: YOLOv8n imgsz=320 (이미지 크기 변경)
# Test 4: YOLOv8n epoch=50 (epoch 변경)
from ultralytics import YOLO
import pandas as pd
import os

results_dir = '/home/ubuntu/results'
os.makedirs(results_dir, exist_ok=True)

def get_results(run_name):
    df = pd.read_csv(
        f'/home/ubuntu/car_damage/runs/detect/{run_name}/results.csv')
    last = df.iloc[-1]
    return {
        'test'     : run_name,
        'mAP50'    : round(float(last['metrics/mAP50(B)']), 4),
        'mAP50-95' : round(float(last['metrics/mAP50-95(B)']), 4),
        'precision': round(float(last['metrics/precision(B)']), 4),
        'recall'   : round(float(last['metrics/recall(B)']), 4),
    }

all_results = []

# test 1  
model = YOLO('yolov8n.pt')  # load pretrained model
results = model.train(
    data='/home/ubuntu/car_damage/cardd.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='test1',
    device=0
)
all_results.append(get_results('test1'))

# test 2 
model = YOLO('yolov8s.pt')  # bigger model
results = model.train(
    data='/home/ubuntu/car_damage/cardd.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='test2',
    device=0
)
all_results.append(get_results('test2'))

# test 3
model = YOLO('yolov8n.pt')
results = model.train(
    data='/home/ubuntu/car_damage/cardd.yaml',
    epochs=30,
    imgsz=320,
    batch=16,
    name='test3',
    device=0
)
all_results.append(get_results('test3'))

# test 4 
model = YOLO('yolov8n.pt')
results = model.train(
    data='/home/ubuntu/car_damage/cardd.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='test4',
    device=0
)
all_results.append(get_results('test4'))

# save results
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
df.to_csv(f'{results_dir}/train_summary.csv', index=False)
