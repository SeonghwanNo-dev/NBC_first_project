from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

import Module.processed_dataset as processed_dataset
import Module.experiment_tool as exp_tool
import Module.setSeed as setSeed
import Module.TextPreprocessingPipeline as T_Preprocessor

import Module.classifier as classifier
import Module.white_kd as w_kd
import Module.kd_trainer as kd_trainer

# 1. 모델 아키텍쳐 확인
models = ["kakaocorp/kanana-1.5-8b-base", "klue/roberta-base", "klue/bert-base", "kykim/bert-kor-base", "beomi/kcbert-base", "monologg/koelectra-base-v3-discriminator"]
# 1.1 가중치 없이 설정 파일만 다운로드하고 로드
# config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
for i in models:
    config = AutoConfig.from_pretrained(i, trust_remote_code=True)
    # 1.2 구조 정보 확인
    # 총 레이어 수 (Transformer 블록 수)
    num_layers = config.num_hidden_layers
    # 피처 개수 (Hidden Size, 각 레이어의 출력 차원)
    hidden_size = config.hidden_size
    print(f"{i}, Total Layers (num_hidden_layers): {num_layers}")
    print(f"{i}, Feature Size (hidden_size): {hidden_size}")
    
"""
- Teacher
Total Layers (num_hidden_layers): 32
Feature Size (hidden_size): 4096

- Student
  Total Layers (num_hidden_layers): 12
  Feature Size (hidden_size): 768
"""
teacher_hidden_dimension=4096
student_hidden_dimension=768
teacher_layer_num=32
student_layer_num=12
# 0번째 hindden layer는 아직 트랜스포머를 거치기 전, 막 인코딩 됐을 때
# 따라서 1 to 32(or 12)까지 존재함.
T_hint_layers = [1, 11, 21, 32] 
S_hint_layers = [1, 4, 8, 12]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
setSeed.set_seed(RANDOM_STATE)

# tool 사용하기
PATH_TO_STORE = './results'
PROJECT_NAME = 'White_Knowledge_Distillation'
HW_COUNT = torch.cuda.device_count()
HW_NAME = torch.cuda.get_device_name(0)
CONFIG = {
    "seed": RANDOM_STATE,
    "batch_size": 1, 
    "teacher_models": "kakaocorp/kanana-1.5-8b-base",
    "student_models": "klue/bert-base",
    "teacher_demension": teacher_hidden_dimension,
    "student_dimension": student_hidden_dimension,
    "teacher_layer_num": teacher_layer_num,
    "student_layer_num": student_layer_num,
    "teacher_hint_layers":"1, 11, 21, 32",
    "student_hint_layers":"1, 4, 8, 12",
    "device_name": HW_NAME,
    "device_count": HW_COUNT,
}
e_tool = exp_tool.ExperimentTool(PATH_TO_STORE, PROJECT_NAME, CONFIG)

# dataset 가져오기
base_dir = Path(__file__).resolve().parent
file_path = base_dir/'data/train.csv'
df = pd.read_csv(file_path, encoding='utf-8')
reviews = df['review']
labels = df['label']

T_Preprocessor_instance = T_Preprocessor.TextPreprocessingPipeline()
X_train_processed, y_train = T_Preprocessor_instance.fit_transform(reviews, labels)

e_tool.d_log(X_train_processed, "X_train_processed")

X_train_processed = X_train_processed.tolist()
y_train = y_train.tolist()

# 1. transformer_model <- Teacher Model
t_model_name = "kakaocorp/kanana-1.5-8b-base"
t_model = AutoModelForCausalLM.from_pretrained(
    t_model_name, 
    trust_remote_code=True, 
    device_map="auto",              # 다중 GPU 로드 또는 CPU 오프로딩
    torch_dtype=torch.float16,      # float8 지원 안 해서 float16으로 rollback
    output_hidden_states=True       # KD 피처 추출을 위해 hidden_states 출력을 명시
)
t_tokenizer = AutoTokenizer.from_pretrained(t_model_name)

# CLM 모델은 패딩 토큰 설정이 필요함
if t_tokenizer.pad_token is None:
    t_tokenizer.pad_token = t_tokenizer.eos_token
t_model.eval() # 피처 추출은 평가 모드로 실행
    
# transformer_model <- Student Model
s_model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


# 2. data encoding -> dataset -> DataLoader 생성
# tokenizer로 encoding 생성
train_full_encodings = tokenizer(
        X_train_processed, 
        truncation=True, 
        padding='max_length', 
        max_length= 256
    )
# Dataset 생성
train_dataset = processed_dataset.processed_dataset(train_full_encodings, y_train)
# DataLoader 생성
# 14B 모델에서 batch_size=256은 불가능
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


# 3. Teacher Model에서 hint_layers와 soft_label 추출
w_kd_instance = w_kd.extract_from_teacher(train_loader=train_loader, model=t_model, T_hint_layers=T_hint_layers, e_tool=e_tool)

# 4. 추출된 값으로부터 Student Model 학습하기

'''
KD 학습을 처리할 클래스에는 세 가지 유형의 손실 함수가 필요하다.
1. Feature Loss: Teacher의 Layer 출력(힌트 레이어)과 Student의 Layer 출력을 비교(MSE Loss).
2. Distillation Loss: Teacher의 로짓(Soft Label)과 Student의 로짓을 비교 (Temperature-scaled Cross-Entropy).
3. Student Loss: Student의 로짓과 실제 정답 레이블(Hard Label)을 비교 (Cross-Entropy).


Distillation 전략
1. layer 1의 Feature Loss
2. layer 1 ~ 4 Feature Loss
3. layer 1 ~ 8 Feature Loss
4. layer 1 ~ 12 Feature Loss
5. layer 1 ~ 12 Feature Loss + Distillation Loss
6. layer 1 ~ 12 Feature Loss + Distillation Loss + Student Loss

필요한 것
1. 각각의 loss function
2. feature 차원이 안 맞으므로 W 레이어 붙이기 모듈로 만들어서 붙이기
3. Distillation 전략을 파이프라인으로 구현하기
'''

# clf = classifier.ClassificationHead(hidden_size=student_hidden_dimension, num_labels=4, inner_dim=256)
# # classifier만 직접 전달해주고, projector는 StudentWithProjector에서 스스로 만든다.
# model = w_kd.StudentWithProjector(S_model= s_model, S_hint_layers=S_hint_layers, t_dim=teacher_hidden_dimension, s_dim = student_hidden_dimension, middle_dim = 4096, classifier=clf)

# params_to_learn = list(model.parameters())
# optimizer = optim.AdamW(params=params_to_learn, lr=2e-5)
# criterion = nn.CrossEntropyLoss()
# teacher_soft_labels=0
# t_features=0
# kd_trainer_instance = kd_trainer.Trainer(experiment_tool=e_tool, teacher_soft_labels = teacher_soft_labels, t_features=t_features, StudentWithProjector_model=model, criterion=criterion, optimizer=optimizer, device=DEVICE)

# transformer_layers_to_freeze = [0,1,2,3] # 0 to 11 까지 존재
# projector_layers_to_learn = [0,1]
# alpha = [0, 0.5, 0.5] # loss = alpha[0]*student_loss + alpha[1]*distillation_loss + alpha[2]*feature_loss
# kd_trainer_instance.train(train_loader=train_loader, epoch = 1, transformer_layers_to_freeze=transformer_layers_to_freeze, projector_layers_to_learn=projector_layers_to_learn, alpha=alpha)