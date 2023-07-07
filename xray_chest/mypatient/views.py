from django.shortcuts import render, redirect
from mypatient.models import Patient
from django.shortcuts import render
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.utils import load_img
from tensorflow import keras
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import keras.models
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Input, Lambda, GlobalAveragePooling2D
from keras.models import Model
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
        ])
UPLOAD_DIR='d:/xray_chest/mypatient/static/images/'
# 이미지 추가 전처리
def apply_gaussian_filter(image):
    np_image = image.numpy()  # 텐서를 넘파이 배열로 변환
    if len(np_image.shape) == 3:
        np_image = np_image.transpose(1, 2, 0)  # 차원 변경 (C, H, W) -> (H, W, C)
        filtered_image = cv2.GaussianBlur(np_image, (3, 3), 0)
        filtered_image = filtered_image.transpose(2, 0, 1)  # 차원 변경 (H, W, C) -> (C, H, W)
        return torch.from_numpy(filtered_image)  # 넘파이 배열을 텐서로 변환
    else:
        return image

def home(request):
    try :
        patient_name =request.POST["patient_name"]  #==> # 환자 이름이 이곳으로 전달되도록 변수 추가
    except:
        patient_name = ""
    items = Patient.objects.filter(name__contains = patient_name).order_by("name") #==> #필터 이용 쿼리에서 환자이름 검색함

    # items = Patient.objects.order_by("name") ==> 위로 변경
    return render(
        request, "mypatient/list.html", {"items": items, "mypatient_count": len(items)}
    )


def regi(request):
    if "file1" in request.FILES:
        file = request.FILES["file1"]
        file_name = file._name
        fp = open("%s%s" % (UPLOAD_DIR, file_name), "wb")
        for chunk in file.chunks():
            fp.write(chunk)
            fp.close()
    else:
        file_name = "-"
    if request.method == 'POST':
        print('post')
        name = request.POST['name']
        age = request.POST['age']
        height = request.POST['height']
        weight = request.POST['weight']
        blood_type = request.POST['blood_type']
        last_visit = request.POST['last_visit']
        memo = request.POST['memo']
        picture_url = file_name
        p=Patient(name=name,age=age, height=height,
                weight=weight,blood_type=blood_type,last_visit=last_visit,memo=memo,
                picture_url=picture_url)
        print('p:',p)
        p.save()
        return redirect("/mypatient")
    else:
        return render(request, 'mypatient/main_2.html')

def detail(request):
    id = request.GET["idx"]
    print('id:'+id)
    pat = Patient.objects.get(idx=id)
    print('pat:',pat)
    return render(request, "mypatient/detail.html", {"pat": pat})

def patient_info(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    return render(request, "mypatient/patient_info.html", {"pat": pat})

def update(request):
    idx = request.POST['idx']
    row_src = Patient.objects.get(idx=idx)
    p_url = row_src.picture_url
    if "file1" in request.FILES:
        file = request.FILES["file1"]
        p_url = file._name
        fp = open("%s%s" % (UPLOAD_DIR, p_url), "wb")
        for chunk in file.chunks():
            fp.write(chunk)
            fp.close()
    row_new = Patient(idx=idx,
                      name=request.POST["name"],
                      age=request.POST["age"],
                      height=request.POST["height"],
                      weight=request.POST["weight"],
                      blood_type=request.POST["blood_type"],
                      last_visit=request.POST["last_visit"],
                      memo=request.POST["memo"],
                      picture_url=p_url)
    row_new.save()
    return redirect("/mypatient")


def delete(request):
    Patient.objects.get(idx=request.POST["idx"]).delete()
    return redirect("/mypatient")

def search(request):
    return render( request, "mypatient/search.html")

def receipt(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    return render(request, "mypatient/info2.html", {"pat":pat})

def register(request):
    return render(request, "mypatient/main_2.html")


# 모델을 위한 코드

def Atelectasis(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    class_names = ['Atelectasis', 'Normal']
    image_file = (f"mypatient/static/images/{image}")  # 업로드된 이미지 파일을 가져옵니다.
    image = Image.open(image_file)
    image = image.convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    model = torch.load('mypatient/model/Atelectasis.h5')
    model.to(device)
    model.eval()
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    #return render(request, 'c:/python/myproject/myapp/templates/Atelectasis.html', {'prediction': prediction})
    return render(request, 'mypatient/Atelectasis.html', {'prediction': predicted_label})


def Cardiomegaly(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    model = keras.models.load_model('mypatient/model/Cardiomegaly_model.h5', compile=False)
    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    class_names = ['Normal', 'Cardiomegaly']
    img_size = (299, 299)
    image_file = (f"mypatient/static/images/{image}")
    image = Image.open(image_file)
    image = image.resize(img_size)
    img = image.convert("RGB")
    img = np.array(img)
    img = np.reshape(img, (1, 299, 299, 3))
    predicted_label = class_names[np.argmax(model.predict(img))]
    return render(request, 'mypatient/Cardiomegaly.html', {'prediction': predicted_label})

def Edema(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    class_names = ['Edema', 'Normal']
    # 모델 정의 및 불러오기
    model = torch.load('mypatient/model/Edema.h5')
    model.to(device)
    model.eval()
    # 이미지 파일 불러오기
    image = Image.open(image_file)  # 이미지 파일 경로 설정
    image = image.convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Edema.html', {'prediction': predicted_label})

def Effusion2(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    class_names = ['Effusion', 'Normal']
    # 모델 정의 및 불러오기
    model = torch.load('mypatient/model/Effusion.h5')
    model.to(device)
    model.eval()
    # 이미지 파일 불러오기
    image = Image.open(image_file)  # 이미지 파일 경로 설정
    image = image.convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
     # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Effusion2.html', {'prediction': predicted_label})

def Fibrosis(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    class_names = ['Fibrosis', 'Normal']
    # 모델 정의 및 불러오기
    model = torch.load('mypatient/model/Fibrosis.h5')
    model.to(device)
    model.eval()
    # 이미지 파일 불러오기
    image = Image.open(image_file)  # 이미지 파일 경로 설정
    image = image.convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Fibrosis.html', {'prediction': predicted_label})

def Pneumonia(request):
    class_names = ['Normal', 'Pneumonia']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 정의 및 불러오기
    model = torch.load('mypatient/model/Pneumonia3.h5')
    model.to(device)
    model.eval()
    # 이미지 불러오기
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    image = Image.open(image_file)  # 이미지 파일 경로 설정
    image = image.convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Pneumonia.html', {'prediction': predicted_label})


def Pneumothorax(request):
    class_names = ['Normal', 'Pneumothorax']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 정의 및 불러오기
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    model = torch.load('mypatient/model/Pneumothorax2.h5')
    model.to(device)
    model.eval()
    # 이미지 불러오기
    image = Image.open(image_file)  # 이미지 파일 경로 설정
    image = image.convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Pneumothorax.html', {'prediction': predicted_label})


def Tuberculosis(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    test_data = []
    # 모델 정의 및 불러오기
    model = keras.models.load_model('mypatient/model/Tuberculosis_model.h5')
    # 이미지 파일 불러오기
    img = Image.open(image_file)
    # Check if the image is grayscale (L mode)
    if img.mode == 'L':
        # Convert grayscale image to RGB
        img_rgb = img.convert('RGB')
    else:
        # For non-grayscale images, use the original image
        img_rgb = img
    img = np.array(img_rgb)
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img / 255
    test_data.append(img)
    test_data1 = np.array(test_data)
    a = model.predict(test_data1)
    # 예측 결과 확인
    if a[0][0] > a[0][1]:
        predicted_label = 'Normal'
    else:
        predicted_label = 'Tuberculosis'
    return render(request, 'mypatient/Tuberculosis.html', {'prediction': predicted_label})

def Multiclassification(request):
    id = request.GET["idx"]
    pat = Patient.objects.get(idx=id)
    # 클래스명
    class_names = ['Atelectasis', 'Cardiomegaly', 'Edema',
                   'Effusion', 'Fibrosis', 'Normal',
                   'Pneumonia', 'Pneumothorax', 'Tuberculosis']
    # 데이터 로딩 및 전처리
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: apply_gaussian_filter(x)),  # 가우시안 필터링 적용
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 모델 정의 및 불러오기
    model = torch.load('mypatient/model/MultiClassification.h5')
    model.to(device)
    model.eval()
    # 이미지 불러오기
    image = pat.picture_url
    image_file = (f"mypatient/static/images/{image}")
    image = Image.open(image_file)
    image = image.convert('RGB')
    input_image = transform(image).unsqueeze(0).to(device)
    # 예측 수행
    with torch.no_grad():
        output = model(input_image)
    # 예측 결과 확인
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]
    return render(request, 'mypatient/Total.html', {'prediction': predicted_label})
