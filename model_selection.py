from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Module.classifier as classifier
import Module.trainer as trainer
import Module.processing_data as processing_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = ["klue/roberta-base", "klue/bert-base", "kykim/bert-kor-base", "beomi/kcbert-base", "monologg/koelectra-base-v3-discriminator"]

# Main Function
for i in models:
    
    # 1. transformer_model, tokenizer, classifier_head
    transformer_model = AutoModel.from_pretrained(i)
    tokenizer = AutoTokenizer.from_pretrained(i)
    classification_head = classifier.ClassificationHead(hidden_size=transformer_model.config.hidden_size, NUM_LABELS=4).to(DEVICE)
    
    # 2. data encoding 및 DataLoader 생성 (모든 리뷰를 한 번에 처리)
    file_path = 'data/train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')
    reviews = df['review'].astype(str).tolist()
    labels = df['label']

    full_encodings = tokenizer(
        reviews, 
        truncation=True, 
        padding='max_length', 
        max_length= 256
    )
    
    # Dataset과 DataLoader 생성
    train_dataset = processing_data.processed_dataset(full_encodings, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )
    
    # 3. Optimizer 설정
    params_to_learn = list(classification_head.parameters())
    optimizer = optim.AdamW(params=params_to_learn, lr=2e-5)
    
    # 4. Trainer 인스턴스 생성 및 훈련 시작
    criterion = nn.CrossEntropyLoss()
    trainer_instance = trainer.Trainer(
        freeze_transformer=True, 
        freeze_classifier=False, # 분류기 헤드를 학습시키기 위해 False로 설정
        transformer_model=transformer_model, 
        classification_head=classification_head, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=DEVICE
    )
    
    trainer_instance.run_training(train_loader, train_loader, num_epochs=10 ) # 편의상 test_loader 대신 train_loader 사용
    
    
    