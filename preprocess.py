#preprocess.py
#COCO annotation을 YOLO 형식으로 변환
import os
import json
from pathlib import Path

base = '/home/ubuntu/CarDD_release/CarDD_COCO'
label_dir = '/home/ubuntu/CarDD_release/CarDD_COCO/labels'

# check dataset
for split in ['train2017', 'val2017', 'test2017']:
    path = f'{base}/{split}'
    count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
    print(f'{split}: {count} images')

# check annotation
with open(f'{base}/annotations/instances_train2017.json') as f:
    data = json.load(f)
print('categories:', data['categories'])
print('total images:', len(data['images']))
print('total annotations:', len(data['annotations']))

# COCO to YOLO conversion
# COCO : [x_min, y_min, width, height]
# YOLO : [class x_center y_center width height] normalized 0~1
def coco_to_yolo(json_path, save_dir):
    with open(json_path) as f:
        data = json.load(f)
    os.makedirs(save_dir, exist_ok=True)
    #image id로 이미지 정보 찾
    img_info = {img['id']: img for img in data['images']}
    # 이미지별로 annotation 묶기
    ann_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)
    all_files = {}
    for img_id, anns in ann_by_img.items():
        img = img_info[img_id]
        W, H = img['width'], img['height']
        lines = []
        for ann in anns:
            x_min, y_min, w, h = ann['bbox']
            x_center = (x_min + w / 2) / W
            y_center = (y_min + h / 2) / H
            w_norm = w / W
            h_norm = h / H
            cls = ann['category_id'] - 1
            lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )
        # 이미지와 같은 이름의 .txt 라벨 파일 생성
        label_name = Path(img['file_name']).stem + '.txt'
        all_files[label_name] = '\n'.join(lines)
    for label_name, content in all_files.items():
        with open(os.path.join(save_dir, label_name), 'w') as f:
            f.write(content)
    print(f'done: {len(ann_by_img)} labels saved to {save_dir}')

coco_to_yolo(
    f'{base}/annotations/instances_train2017.json',
    f'{label_dir}/train2017')
coco_to_yolo(
    f'{base}/annotations/instances_val2017.json',
    f'{label_dir}/val2017')
coco_to_yolo(
    f'{base}/annotations/instances_test2017.json',
    f'{label_dir}/test2017')
