# val.py
# 학습된 4개 모델을 검증 데이터(val2017)로 평가
# mAP50, mAP50-95, Precision, Recall 확인
from ultralytics import YOLO

models_list = [
    'test1',
    'test2',
    'test3',
    'test4',
]

for run_name in models_list:
    print(f'\n--- {run_name} ---')
    model = YOLO(f'runs/detect/{run_name}/weights/best.pt')
    results = model.val(
        data='cardd.yaml',
        imgsz=640,
        batch=16,
        device=0
    )
    print('mAP50:', results.box.map50)
    print('mAP50-95:', results.box.map)
    print('Precision:', results.box.mp)
    print('Recall:', results.box.mr)
