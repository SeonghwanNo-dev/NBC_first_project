from tqdm import tqdm
import torch
import torch.nn as nn
from. white_kd import distillation_loss

class Trainer:
    def __init__(self, experiment_tool, teacher_soft_labels, t_features, StudentWithProjector_model, criterion, optimizer, device):
        self.experiment_tool = experiment_tool
        self.teacher_soft_labels = teacher_soft_labels
        self.t_features = t_features

        self.model = StudentWithProjector_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mse_loss = nn.MSELoss()

        # 특징 추출기(트랜스포머)의 가중치 동결
        for param in self.StudentWithProjector_model.parameters():
            param.requires_grad = True
            
        self.model.to(device)
        
    def Transformer_layers_to_unfreeze(self, transformer_layers_to_freeze):
        # 지정된 Transformer 인코더 레이어만 동결 해제 (Unfreeze)
        if transformer_layers_to_freeze:
            # 인코더 레이어는 'bert.encoder.layer.[i]'와 같은 패턴을 가집니다.
            # (RoBERTa의 경우 model.roberta.encoder.layer.[i] 등)
            for layer_idx in transformer_layers_to_freeze:
                # 패턴 매칭 (이름에 레이어 인덱스가 포함된 모든 파라미터 찾기)
                layer_pattern = f'.layer.{layer_idx}.' 
                
                for name, param in self.model.named_parameters():
                    if layer_pattern in name:
                        param.requires_grad = False
                        print(f"동결 해제: {name}") # 디버깅용
                        
    def train(self, train_loader, epoch, transformer_layers_to_freeze, projector_layers_to_learn, alpha):
        
        # self.classification_head.train() <- classification은 풀어야 함. 구현해야함
        self.Transformer_layers_to_unfreeze(transformer_layers_to_freeze)

        total_loss = 0        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # with torch.no_grad() if self.freeze_transformer else torch.enable_grad():
            # 부분 동결 시 no_grad()를 사용하지 않고 
            # param.requires_grad 설정을 통해 PyTorch의 자동 미분 시스템에게 책임을 넘겨야 한다.
            model_output = self.model(
                input_ids=batch['input_ids'].to(self.device), 
                attention_mask=batch['attention_mask'].to(self.device), 
                return_dict=True
            )
            
            outputs = model_output['logits']
            student_soft_labels = model_output['student_soft_labels']
            kd_features = model_output['kd_features']
            
            labels = batch['labels'].to(self.device)
            teacher_soft_labels_batch = self.teacher_soft_labels[batch*batch_idx:batch*(batch_idx+1)]
            
            student_loss = self.criterion(outputs, labels)
            distillation_loss = distillation_loss(student_logits=student_soft_labels, teacher_logits=teacher_soft_labels_batch)
            feature_loss = 0.0            
            for i in range(len(projector_layers_to_learn)):
                teacher_features_batch = self.t_features[i][batch*batch_idx:batch*(batch_idx+1)]
                feature_loss += self.criterion(kd_features[i], teacher_features_batch)
            feature_loss /= len(projector_layers_to_learn)
            
            loss = alpha[0]*student_loss + alpha[1]*distillation_loss + alpha[2]*feature_loss
            
            # 3. 역전파 및 업데이트 
            loss.backward()
            self.optimizer.step()
            
            log = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "student_train_loss": student_loss.item(),
                "distillation_train_loss": distillation_loss.item(),
                "feature_train_loss": feature_loss.item(),
                "weighted_sum_loss": loss.item(),
                "train_loss/one_epoch": "N", 
                "test_acc": "N"
            }
            self.experiment_tool.log(log)
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
            train_loss = self.train(train_loader, epoch)
            test_accuracy = self.test(test_loader)
            
            train_losses.append(train_loss)
            test_accuracies.append(test_accuracy)
            
            log = {
                "epoch": epoch,
                "batch_idx": "N",
                "student_train_loss": "N",
                "distillation_train_loss": "N",
                "feature_train_loss": "N",
                "weighted_sum_loss": "N",
                "train_loss/one_epoch": train_loss, 
                "test_acc": test_accuracy
            }
            self.experiment_tool.log(log)
        return train_losses, test_accuracies