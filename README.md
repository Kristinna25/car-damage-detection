# Car Damage Detection (자동차 손상 탐지)

YOLOv8 기반 자동차 손상 탐지 프로젝트

**이름:** 수 메이 푸 진
**학번:** 202647027
**과목:** 컴퓨터 비전

---

## 문제 정의

자동차 손상 확인은 중고차 검사, 보험 처리, 정비 과정에서 중요하다.
사람이 직접 확인하면 시간이 오래 걸리고 작은 손상을 놓칠 수 있다.
YOLOv8 모델을 사용해 자동차 이미지에서 손상 위치와 종류를 자동으로
탐지하는 시스템을 구현했다.

탐지 클래스 (6개): dent, scratch, crack, glass shatter, lamp broken, tire flat

---

## 데이터셋

- **이름:** CarDD (Car Damage Dataset)
- **source:** https://cardd-ustc.github.io/
- **구성:** train 2,816장 / val 810장 / test 374장
- **annotation:** COCO format을 YOLO format으로 전처리

---

## 모델 및 실험 설계

YOLOv8n / YOLOv8s를 사용했고, pretrained weight 기반 transfer learning 진행.
4가지 설정으로 학습 후 성능 비교:

| Test | 모델 | Epoch | imgsz | 변경 사항 |
|------|------|-------|-------|---------|
| test1 | YOLOv8n | 30 | 640 | baseline |
| test2 | YOLOv8s | 30 | 640 | 모델 변경 |
| test3 | YOLOv8n | 30 | 320 | 이미지 크기 변경 |
| test4 | YOLOv8n | 50 | 640 | epoch 변경 |

---

## 결과 (Validation, val2017 810장)

| Test | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|---------|-----------|--------|
| test1 | 0.700 | 0.551 | 0.719 | 0.668 |
| test2 | **0.730** | **0.570** | **0.778** | **0.685** |
| test3 | 0.490 | 0.280 | 0.480 | 0.522 |
| test4 | 0.713 | 0.560 | 0.754 | 0.680 |

**최고 성능: test2 (YOLOv8s)**

최종 test 데이터(374장)에서 mAP50 = 0.724 달성

---

## 폴더 구조
car_damage/
├── cardd.yaml       # 데이터셋 설정
├── preprocess.py    # COCO → YOLO 변환
├── train.py         # 4가지 실험 학습
├── val.py           # 모델 검증
├── test.py          # 최종 탐지 테스트
├── result_1~6.jpg   # 탐지 결과 이미지
└── runs/detect/     # 학습 로그 및 결과

---

## 실행 방법

```bash
python preprocess.py   # 데이터 전처리
python train.py        # 4가지 모델 학습
python val.py           # 모델 검증
python test.py          # 탐지 테스트
```

---

## 사용 환경

- AWS EC2 (g4dn.xlarge, Tesla T4 GPU)
- Python 3.13, PyTorch, Ultralytics YOLOv8

