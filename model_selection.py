from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import Module.classifier as classifier
import Module.trainer as trainer
import Module.processing_data as processing_data
import Module.experiment_tool as exp_tool

# tool 사용하기
PATH_TO_STORE = './NBC_first_project/results'
PROJECT_NAME = 'model_selection_v1'
HW_COUNT = torch.cuda.device_count()
HW_NAME = torch.cuda.get_device_name(0)
CONFIG = {
    "lr": 2e-5, 
    "batch_size": 256, 
    "num_epochs": 10,
    "models": "klue/roberta-base, klue/bert-base, kykim/bert-kor-base, beomi/kcbert-base, monologg/koelectra-base-v3-discriminator",
    "device_name": HW_NAME,
    "device_count": HW_COUNT,
}
e_tool = exp_tool.ExperimentTool(PATH_TO_STORE, PROJECT_NAME, CONFIG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = ["klue/roberta-base", "klue/bert-base", "kykim/bert-kor-base", "beomi/kcbert-base", "monologg/koelectra-base-v3-discriminator"]

base_dir = Path(__file__).resolve().parent
file_path = base_dir/'data/train.csv'
df = pd.read_csv(file_path, encoding='utf-8')
reviews = df['review']
labels = df['label']

# dataset만 저장, 가중치는 저장 X(다시 사용할 일 없으므로)
e_tool.d_log(reviews, "input")
e_tool.d_log(labels, "label")

reviews = reviews.astype(str).tolist()



# Main Function
for i in models:
    
    # 1. transformer_model, tokenizer, classifier_head
    transformer_model = AutoModel.from_pretrained(i)
    tokenizer = AutoTokenizer.from_pretrained(i)
    classification_head = classifier.ClassificationHead(hidden_size=transformer_model.config.hidden_size, num_labels=4).to(DEVICE)
    
    # 2. data encoding 및 DataLoader 생성 (모든 리뷰를 한 번에 처리)
    full_encodings = tokenizer(
        reviews, 
        truncation=True, 
        padding='max_length', 
        max_length= 256
    )
    
    # Dataset과 DataLoader 생성
    total_dataset = processing_data.processed_dataset(full_encodings, labels)
    train_ratio = 0.8
    train_loader, test_loader = processing_data.divide_into_TrainAndTest(total_dataset, train_ratio)
    
    # 3. Optimizer 설정
    params_to_learn = list(classification_head.parameters())
    optimizer = optim.AdamW(params=params_to_learn, lr=2e-5)
    
    # 4. Trainer 인스턴스 생성 및 훈련 시작
    criterion = nn.CrossEntropyLoss()
    trainer_instance = trainer.Trainer(
        experiment_tool = e_tool,
        freeze_transformer=True, 
        freeze_classifier=False, # 분류기 헤드를 학습시키기 위해 False로 설정
        transformer_model=transformer_model, 
        classification_head=classification_head, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=DEVICE
    )
    
    trainer_instance.run_training(train_loader, test_loader, num_epochs=10 )
    
