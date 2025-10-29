import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .Hint_layer_projector import HintProjector

def extract_from_teacher(train_loader, model, T_hint_layers, e_tool):
    # 각 배치의 피처 리스트 (teacher_features)와 소프트 레이블 리스트 (teacher_soft_labels)를 담을 리스트
    teacher_features_batches = []
    teacher_soft_labels_batches = []
    
    total_correct = 0
    total_samples = 0

    model.eval() # 모델을 반드시 평가 모드로 설정
    
    with torch.no_grad():
        for batch in train_loader:
            # 데이터 GPU로 이동 (device_map="auto"의 경우 모델의 디바이스를 따름)
            # model.device를 사용하면, 모델이 로드된 디바이스(들)를 자동으로 따라갑니다.
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            # 모델 포워드 (Logits 및 Hidden States 추출)
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            # 1. 소프트 레이블 (Logits) 추출 및 저장
            # CausalLM (HyperCLOVAX)은 최종 분류를 위해 시퀀스의 마지막 토큰 로짓만 사용
            # [BATCH_SIZE, VOCAB_SIZE]
            soft_labels = outputs.logits[:, -1, :] 
            teacher_soft_labels_batches.append(soft_labels.cpu())
            
            # 2. 힌트 레이어 피처 추출 및 저장 (outputs.hidden_states는 튜플)
            hidden_states = outputs.hidden_states
            
            # T_hint_layers의 인덱스에 해당하는 레이어의 피처만 추출하여 저장
            # 추출된 피처는 CPU로 이동
            hint_features = [hidden_states[i].cpu() for i in T_hint_layers]
            teacher_features_batches.append(hint_features)
            
            # 3. 정확도 측정 로직 추가
            predictions = soft_labels.argmax(dim=-1)
            target_labels = labels
            # 예측과 정답 비교
            correct_predictions = (predictions == target_labels).sum().item()
            total_correct += correct_predictions
            total_samples += target_labels.size(0)
            
    # --- 데이터 정리 및 저장 ---
    
    # 1. 전체 데이터셋에 대한 정확도 계산
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    log = {"teacher_total_accuracy": total_accuracy}
    e_tool.log(log)

    # 2. 소프트 레이블 합치기: 모든 배치의 소프트 레이블 텐서를 하나로 합칩니다.
    final_soft_labels = torch.cat(teacher_soft_labels_batches, dim=0)

    # 3. 피처 합치기: T_hint_layers의 각 레이어별로 모든 배치의 피처를 합칩니다.
    # 최종 결과: [T_hint_layers의 길이] x [전체 데이터 수] x [시퀀스 길이] x [히든 사이즈] 텐서 목록
    final_teacher_features = []
    num_hint_layers = len(T_hint_layers)
    
    for i in range(num_hint_layers):
        # 각 힌트 레이어 (i)에 대해 모든 배치의 텐서를 추출하여 합침
        layer_features = torch.cat([batch_features[i] for batch_features in teacher_features_batches], dim=0)
        final_teacher_features.append(layer_features)
        
    # 4. e_tool을 사용하여 저장
    # final_teacher_features는 리스트
    
    # 리스트 자체를 저장하거나, 각 레이어별로 저장하도록 구현할 수 있습니다.
    e_tool.f_log(final_teacher_features, "teacher_features_all_layers") # 리스트 형태로 저장 시
    e_tool.w_log(final_soft_labels, "soft_label") # Tensor 형태로 저장

    print(f"✅ 소프트 레이블 추출 완료. 전체 크기: {final_soft_labels.shape}")
    print(f"✅ 피처 추출 완료. {len(final_teacher_features)}개 레이어 저장.")
    
    
    
    
    
def distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Soft Label을 이용한 Distillation Loss (Temperature-scaled Cross-Entropy)를 계산합니다.
    
    Args:
        teacher_logits (torch.Tensor): Teacher Model의 최종 로짓.
        student_logits (torch.Tensor): Student Model의 최종 로짓.
        T (float): Temperature (온도) 값.
        
    Returns:
        torch.Tensor: Distillation Loss 값.
    """
    # 1. Temperature 스케일링 적용
    # Softmax를 적용하기 전 로짓을 T로 나누어 부드러운 분포를 만듭니다.
    T_logits = teacher_logits / T
    S_logits = student_logits / T
    
    # 2. Teacher의 Soft Label (확률 분포) 생성 (log_softmax를 사용하여 안정성 확보)
    # log_softmax는 log(softmax(x))를 계산합니다.
    p_t = nn.functional.log_softmax(T_logits, dim=-1)
    
    # 3. KL Divergence 계산: D_KL(P_Teacher || P_Student)
    # nn.KLDivLoss는 log(P)와 Q를 입력으로 받습니다. (log(P_Student)와 P_Teacher를 사용)
    # reduction='batchmean'은 PyTorch Distillation에서 권장되는 방식입니다.
    kl_loss = nn.functional.kl_div(
        nn.functional.log_softmax(S_logits, dim=-1),  # log(Q) = log(P_Student)
        nn.functional.softmax(T_logits, dim=-1),      # P = P_Teacher (Soft Label)
        reduction='batchmean'
    )
    
    # T^2를 곱하여 KL Div Loss의 크기를 정규화합니다.
    return kl_loss * (T * T)



    

class StudentWithProjector(nn.Module):
    def __init__(self, S_model, S_hint_layers: list, t_dim: int, s_dim: int, middle_dim: int, classifier):
        super().__init__()
        # 1. 학생 모델 (예시: BERT Encoder)
        # Hugging Face 모델 사용 시 output_hidden_states=True 필수
        self.S_model = S_model
        
        # 2. 피처를 추출할 레이어 인덱스
        self.S_hint_layers = S_hint_layers # [1, 4, 8, 12]
        
        # 3. 각 추출 레이어마다 별도의 Projector 정의 (ModuleList 사용)
        self.projectors = nn.ModuleList([
            HintProjector(s_dim, t_dim, middle_dim)
            for _ in range(len(self.S_hint_layers))
        ])
        self.classifier = classifier
        
    
    # forward 함수의 역할 (순전파)  
    # 입력 데이터를 받아 예측값을 계산하고 반환하는 역할만 수행
    def forward(self, input_ids, attention_mask):
        
        # BERT 모델 실행
        # output_hidden_states=True로 설정했으므로 hidden_states를 반환함
        outputs = self.S_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # hidden_states는 (Embedding + 12개 레이어 출력) 리스트입니다.
        hidden_states = outputs.hidden_states
        
        # KD 피처 리스트 초기화
        kd_features = []
        
        # 원하는 레이어의 출력을 추출하고 Projector에 통과시킴
        for i, layer_idx in enumerate(self.S_hint_layers):
            # 1. 트랜스포머 블록의 출력(F_S) 추출
            student_feature = hidden_states[layer_idx] # (Batch Size, Sequence Length, S_dim)
            # 2. Projector에 통과시켜 F_KD 획득
            projected_feature = self.projectors[i](student_feature) # (Batch Size, Sequence Length, T_dim)
            kd_features.append(projected_feature)
            
        student_soft_labels = outputs.logits[:, -1, :] 
        cls_feature = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_feature)
        
        
        # 반환: KD Loss 계산에 필요한 피처 리스트와 최종 분류 출력
        return kd_features, student_soft_labels, logits


        
# class white_kd_pipeline:
#     def __init__(self):
#         self.model
#     def run(self):
#         self.model = StudentWithProjector(S_model = , S_hint_layers: list, t_dim: int, s_dim: int, middle_dim: int, classifier)
    