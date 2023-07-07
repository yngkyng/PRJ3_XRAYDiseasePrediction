# 흉부 X-ray를 통한 질병 유무 파악 모델
- 사용 방법
-> X-ray 파일을 다운로드
-> D드라이브에 추가 후, pycharm으로 실행
-> migration, migrate
-> python manage.py runserver

## 데이터 출처
- Chest X-Ray Images (Pneumonia)
/ link : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Tuberculosis (TB) Chest X-ray Database
/ link : https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
- NIH Chest X-rays
/ link : https://www.kaggle.com/datasets/nih-chest-xrays/data
## 사용 언어 및 라이브러리
- Python (Visual Studio Code, Google Colab)
/ packages : numpy, pandas, seaborn, glob, os, shutil, pathlib, matplotlib, sklearn, skimage, tensorflow, torch/torchvision, PIL, cv2
- HTML, CSS, JS, MySQL (Django)
## 진행 과정
### 1. 프로젝트 주제 선정 및 데이터 수집
### 2. 데이터 분류 및 전처리
- NIH 데이터셋 이미지를 Label별로 분류해서 저장(코드)
- NIH 데이터셋 이미지 중 질이 안 좋아 학습에 저해가 될 것 같은 이미지를 제거(수작업), 이미지 언더/오버샘플링(수작업/코드)
### 3. 이미지 분류 모델 생성
#### 3-1. 이진 분류 모델
- 이미지 분류에 특화된 모델을 여러가지 찾아봄.
- 처음에는 CheXNet(DenseNet)으로 시도해 보았으나, 메모리 부족으로 실패.
- 경량화된 모델인 MobileNet을 대신 사용함.
- torchvision.transforms의 Resize, RandomCrop, ToTensor, Noemalize로 이미지 전처리를 행함.
- 손실함수를 CrossEntropyLoss, optimizer를 Adam으로 잡고 Epoch 1로 돌림(Epoch 2 이상에서 과적합 발견).
#### 3-2. 다중 분류 모델
- 이미지 분류에 특화된 모델을 여러가지 찾아봄.
- DenseNet, ResNet, VGGNet, MobileNet, SqueezeNet, ShuffleNet, EfficientNet으로 시도.
- DenseNet, VGGNet은 메모리 부족으로 실패, 나머지 모델들을 학습시켜 정확도를 비교해본 결과 ShuffleNet으로 선정.
- torchvision.transforms의 Resize, RandomRotation, RandomHorizonalFlip, RandomCrop, ColorJitter, ToTensor, Normalize 및 GaussianBlur로 이미지 전처리를 행함.
- 손실함수를 CrossEntropyLoss, optimizer를 Adam으로 잡고 Epoch 30으로 돌림.
#### 3-3. 결과
이진 분류 모델 Atelectasis 68.5%, Cardiomegaly 75.8%, Edama 83.5%, Effusion 77.5%, Fibrosis 81.5%, Pneumonia 78.1%, Pneumothorax 95%, Tuberculosis 98.6%, 다중 분류 모델 54%의 정확도를 보임.
### 4. X-ray 이미지 진단 웹사이트 구현(Django)
xray_chest
## 참고 자료
- 모델들 저장 위치 :
https://drive.google.com/drive/folders/1lZXUQ-BEtSo6AYv0JqC4qfrtdDbEdHhe?usp=sharing
- HTML 구성 :
https://docs.google.com/presentation/d/1b7BcH8VtjdQAP-fsywoN1aUkzYDJy08pcymOeZ39gbs/edit?usp=sharing
- 발표 PPT :
https://docs.google.com/presentation/d/1Er2joxUuZdLO1go-fyhQKWXsV3WoEb7-0mJmTqM_Zh8/edit?usp=sharing
