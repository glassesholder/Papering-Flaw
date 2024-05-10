## 🔨 Dacon 도배 하자 유형 분류 AI경진대회

### ✅ 요약
1. 주최 : 한솔데코
2. 주관 : 데이콘
3. 설명 : 총 19가지의 도배 하자 유형을 분류하는 AI 모델을 개발하여야 한다. 도배 하자 유형은 다음과 같다.<br>
(가구수정, 걸레받이수정, 곰팡이, 꼬임, 녹오염, 들뜸, 면불량, 몰딩수정, 문틀창틀수정, 반점, 석고수정, 오염, 오타공, 울음, 이음부불량, 터짐, 틈새과다, 피스, 훼손)
4. 팀원 : 이신영, 이효준
5. 결과<br>
**EfficientNet_b4** 사전 학습 모델을 불러와 추가 레이어를 쌓은 뒤 학습시킨 결과, **F1score 0.539**를 기록하였다.

---

### 📁 데이터셋
**1. train [폴더]**
- 19개의 Class 폴더 내 png 파일 존재

**2. test [폴더]**
- 평가용 데이터셋
- 000.png ~ 791.png

**3. test.csv [파일]**
- id : 평가 샘플 고유 id
- img_path : 평가 샘플의 이미지 파일 경로

**4. sample_submission.csv [제출양식]**
- id : 평가 샘플 고유 id  
- label : 예측한 도배 하자 Class

---

### 🔗 개발환경 세팅
**1. 설치파일**
- Python 3.12.0
- Visual Studio Code
- Git

**2. 사용방법**
- vscode를 열어서 terminal을 열어주세요.
- terminal 에서 다음 명령문을 입력해 파일을 받아주세요.<br>
  ```$ git clone --single-branch --branch main https://github.com/2shin0/Papering-Flaw.git```
- 내려받은 폴더로 들어가서 가상환경을 만들어 실행해주세요.<br>
  ```$ python -m venv (원하는 가상환경 이름)```
- 가상환경 안에서 다음 명령문을 통해 필요한 라이브러리를 설치해주세요.<br>
  ```$ pip install -r requirements.txt```
- 이제 다음 명령문을 입력해 학습된 모델 및 test 데이터셋 예측결과 csv를 받을 수 있습니다.<br>
  ```$ python main.py```

**3. requirements.txt**
```
pandas==2.2.2
opencv-python==4.9.0.80
imbalanced-learn==0.12.2
torch
torchvision
torchaudio
albumentations
tqdm
```

---

### 📊 데이터 전처리 (processing.py)
**1. 이미지 경로 변경**
- 경로 인식을 위해 train data의 img_path 변경 : '\\' → '/'
- 경로 인식을 위해 test data의 img_path 변경 : './' → './data/'

**2. 한글 경로를 읽기 위한 encoding & decoding**

cv2가 한글 경로를 인식하지 못하는 문제 해결을 위해서 이미지를 numpy 배열로 우선 encoding 후, 변환된 배열을 다시 decoding하여 이미지로 변환했다.
```python
img_array = np.fromfile(img_path, np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

**3. 클래스 불균형 문제 탐구**

![image](https://github.com/2shin0/Papering-Flaw/assets/150658909/b635cb9d-80a7-4c24-8dac-a043df254906)

클래스 불균형이 심해 데이터 under/oversampling 필요성을 인식했다. 400개 이상의 데이터를 가진 클래스는 비복원 추출로 400개 추출했으며, 50개 이하의 데이터를 가진 클래스는 복원 추출로 50개를 추출했다. 그 밖에도 다양한 방법을 시도했으나 그대로 학습시키는 것보다 성능이 낮아지는 것을 확인했다. 따라서 최종 모델 학습은 클래스 불균형 해소를 하지 않고 진행했다.

---

### 📈 모델 선정 및 학습 (model.py, model_train.py)
**1. 모델 선정 : EfficientNet_b4**<br>

![image](https://github.com/2shin0/Papering-Flaw/assets/150658909/2aca9c2d-b12c-41e4-bd52-f7560841c3bf)

<EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, 2020, Mingxing Tan & Quoc V. Le><br>

EfficientNet은 이미지 분류 작업에 있어서 적은 파라미터 수에도 매우 효율적이고 우수한 성능을 보이는 딥러닝 모델이다. 네트워크의 깊이, 너비, 해상도의 크기를 조절하여 최적의 모델 구조를 찾아내는 compound scaling 방법을 적용하여 최소한의 파라미터로 높은 성능을 낸다. 위 결과를 바탕으로 EfficientNet b0~b7 중, 파라미터 수 대비 정확도를 고려한 EfficientNet b4를 사전 학습 모델로 선정했다.

**2. label 형식 int → long 변환**

```python
labels = labels.long().to(device)
```

설계한 코드에서는 라벨을 long 형식으로 변환하고 있는 것을 확인할 수 있다. 모델이 예상한 클래스 확률과 실제 클래스를 비교할 때 타입을 일치시키기 위해서다. PyTorch의 nn.CrossEntropyLoss() 손실 함수는 라벨을 long 형식으로 받기 때문에 정수 형식의 라벨을 long 형식으로 변환하여 손실 함수에 전달해야 했다. CUDA를 사용하는 경우에는 모든 텐서가 동일한 타입을 가져야 하므로, CUDA로 데이터를 이동시키기 전에 라벨을 long 형식으로 변환하고 이동시켜야 한다.

---

### ✅ 결과
**👍리더보드(PUBLIC) : 275/1152(등)**

![image](https://github.com/2shin0/Papering-Flaw/assets/150658909/2f9504f0-a843-4bbf-a93a-31eae3dcc79c)

**EfficientNet_b4** 사전 학습 모델을 불러와 추가 레이어를 쌓은 뒤 모델을 학습시켰다. 그 결과, **F1score 0.539**를 기록하여 1152팀 중, 275등을 할 수 있었다. 다양한 사전 학습 모델 탐구 및 하이퍼 파라미터 튜닝을 바탕으로 성능 개선을 할 예정이다.

---
#### 🙌Thank you for reading our README.md🙌
