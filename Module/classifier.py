import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, inner_dim=256):
        super().__init__()
        
        # 1. 중간 비선형 레이어 (보통 입력 크기보다 작거나 같게 설정)
        self.dense = nn.Linear(hidden_size, inner_dim) # [768] -> [256] (예시)
        
        # 2. 최종 분류 레이어
        self.classifier = nn.Linear(inner_dim, num_labels) # [256] -> [4]
        
        # 3. 드롭아웃 및 활성화 함수
        self.dropout = nn.Dropout(0.1) 
        self.activation = nn.GELU() # BERT/RoBERTa 계열에서 주로 사용되는 활성화 함수

    def forward(self, cls_vector):
        # cls_vector: (batch_size, hidden_size)
        
        # 1. 드롭아웃 적용
        cls_vector = self.dropout(cls_vector)
        
        # 2. 첫 번째 Dense 및 활성화 함수 통과
        x = self.dense(cls_vector)
        x = self.activation(x)
        
        # 3. 최종 분류 레이어 통과 (로짓 출력)
        logits = self.classifier(x)
        
        return logits
