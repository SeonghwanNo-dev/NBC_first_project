from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, experiment_tool, freeze_transformer, freeze_classifier, transformer_model, classification_head, criterion, optimizer, device):
        self.experiment_tool = experiment_tool
        self.transformer_model = transformer_model
        self.classification_head = classification_head
        self.freeze_transformer = freeze_transformer
        self.freeze_classifier = freeze_classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # 특징 추출기(트랜스포머)의 가중치 동결
        for param in self.transformer_model.parameters():
            param.requires_grad = not self.freeze_transformer 
            
        # 분류기 헤드의 가중치 동결
        for param in self.classification_head.parameters():
            param.requires_grad = not self.freeze_classifier
            
        self.classification_head.to(device)
        self.transformer_model.to(device)

    def train(self, train_loader):
        
        self.classification_head.train()
        if self.freeze_transformer:
            self.transformer_model.eval() 
        else:
            self.transformer_model.train() 

        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()

            with torch.no_grad() if self.freeze_transformer else torch.enable_grad():
                model_output = self.transformer_model(
                    input_ids=batch['input_ids'].to(self.device), 
                    attention_mask=batch['attention_mask'].to(self.device), 
                    return_dict=True
                )
            
            # 1. [CLS] 벡터 추출
            # last_hidden_state의 첫 번째 토큰 (인덱스 0)이 [CLS] 벡터다.
            cls_vector = model_output.last_hidden_state[:, 0, :]
            
            # 2. 분류기 통과 및 손실 계산
            logits = self.classification_head(cls_vector)
            labels = batch['labels'].to(self.device)
            loss = self.criterion(logits, labels)
            
            # 3. 역전파 및 업데이트 
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def test(self, test_loader):
        
        self.classification_head.eval()
        self.transformer_model.eval()
        
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 1. 특징 추출 및 분류기 통과
                model_output = self.transformer_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    return_dict=True
                )
                
                # [CLS] 벡터 추출
                cls_vector = model_output.last_hidden_state[:, 0, :]
                
                # 2. 분류기 통과 및 예측
                logits = self.classification_head(cls_vector)
                predictions = torch.argmax(logits, dim=1)
                
                # 정확도 계산
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
        accuracy = correct_predictions / total_samples
        return accuracy
    
    def run_training(self, train_loader, test_loader, num_epochs):
        train_losses = []
        test_accuracies = []

        print(f"--- 훈련 시작: Epochs={num_epochs} ---")

        for epoch in range(num_epochs):
            train_loss = self.train(train_loader)
            test_accuracy = self.test(test_loader)
            
            train_losses.append(train_loss)
            test_accuracies.append(test_accuracy)
            
            log = {
            "epoch": epoch,
            "train_loss": train_loss, 
            "test_acc": test_accuracy
            }
            self.experiment_tool.log(log)
        return train_losses, test_accuracies