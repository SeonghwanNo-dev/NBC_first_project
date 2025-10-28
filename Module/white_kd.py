import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import Hint_layer_projector

def extract_from_teacher(train_loader, model, T_hint_layers, e_tool):
    # 각 배치의 피처 리스트 (teacher_features)와 소프트 레이블 리스트 (teacher_soft_labels)를 담을 리스트
    teacher_features_batches = []
    teacher_soft_labels_batches = []

    model.eval() # 모델을 반드시 평가 모드로 설정
    
    with torch.no_grad():
        for batch in train_loader:
            # 데이터 GPU로 이동 (device_map="auto"의 경우 모델의 디바이스를 따름)
            # model.device를 사용하면, 모델이 로드된 디바이스(들)를 자동으로 따라갑니다.
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
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
            
    # --- 데이터 정리 및 저장 ---

    # 1. 소프트 레이블 합치기: 모든 배치의 소프트 레이블 텐서를 하나로 합칩니다.
    final_soft_labels = torch.cat(teacher_soft_labels_batches, dim=0)

    # 2. 피처 합치기: T_hint_layers의 각 레이어별로 모든 배치의 피처를 합칩니다.
    # 최종 결과: [T_hint_layers의 길이] x [전체 데이터 수] x [시퀀스 길이] x [히든 사이즈] 텐서 목록
    final_teacher_features = []
    num_hint_layers = len(T_hint_layers)
    
    for i in range(num_hint_layers):
        # 각 힌트 레이어 (i)에 대해 모든 배치의 텐서를 추출하여 합침
        layer_features = torch.cat([batch_features[i] for batch_features in teacher_features_batches], dim=0)
        final_teacher_features.append(layer_features)
        
    # 3. e_tool을 사용하여 저장
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

def set_training_layers(model: nn.Module, layers_to_unfreeze: List[int] = None):
    """
    Student Model의 특정 레이어(블록)만 학습 가능하도록 설정합니다 (Fine-tuning Layering).
    
    Args:
        model (nn.Module): Student Classifier 모델 인스턴스.
        layers_to_unfreeze (List[int]): 학습을 허용할 Transformer 인코더 레이어의 인덱스 목록 (예: [9, 10, 11])
    """
    
    # 1. 모든 파라미터를 기본적으로 동결(Freeze)
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 2. 분류기 헤드(Classifier Head) 파라미터는 항상 학습 허용
    # AutoModelForSequenceClassification의 분류기 레이어는 보통 'classifier'에 속합니다.
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True

    # 3. 지정된 Transformer 인코더 레이어만 동결 해제 (Unfreeze)
    if layers_to_unfreeze:
        # 인코더 레이어는 'bert.encoder.layer.[i]'와 같은 패턴을 가집니다.
        # (RoBERTa의 경우 model.roberta.encoder.layer.[i] 등)
        for layer_idx in layers_to_unfreeze:
            # 패턴 매칭 (이름에 레이어 인덱스가 포함된 모든 파라미터 찾기)
            layer_pattern = f'.layer.{layer_idx}.' 
            
            for name, param in model.named_parameters():
                if layer_pattern in name:
                    param.requires_grad = True
                    print(f"동결 해제: {name}") # 디버깅용
                    
                    
                
class white_kd_pipeline:
    def __init__(self, model, labels, hint_layer_projector):
        self.model = model
        
        self.criterion = nn.CrossEntropyLoss()
        self.s_output
        self.labels = labels
        
        self.s_logits
        self.t_logits
        
        self.hint_layer_projector = hint_layer_projector
        
    def get_student_loss(self):
        return self.criterion(self.s_output['logits'], self.labels)
    
    def get_distillation_loss(self):
        return distillation_loss(teacher_logits=self.t_logits, student_logits=self.s_logits, T=2)
    
    def get_feature_loss(self):
        total_feature_loss = torch.tensor(0.0)
        num_hint_pairs = len(self.t_hint_layers)
        
        # T와 S의 힌트 레이어 쌍을 순회하며 MSE Loss를 누적
        for i, t_layer_idx in enumerate(self.t_hint_layers):
            s_layer_idx = self.s_hint_layers[i]
            t_feature = self.t_output['hidden_states'][t_layer_idx]
            s_feature = self.s_output['hidden_states'][s_layer_idx]
            
            projector = self.hint_layer_projector[str(t_layer_idx)]
            t_feature_projected = projector(t_feature)
            total_feature_loss += self.mse_loss(s_feature, t_feature_projected)
            
        avg_feature_loss = total_feature_loss / num_hint_pairs
        return avg_feature_loss
    
    
    def calculate_total_loss(self, 
                             alpha_feature: float, 
                             alpha_distill: float, 
                             alpha_student: float, 
                             T: float = 2.0) -> torch.Tensor:
        """
        주어진 가중치에 따라 모든 Loss를 합산하여 최종 손실을 계산합니다.
        """
        
        loss_components = {
            'feature': self.get_feature_loss(),
            'distillation': self.get_distillation_loss(T=T),
            'student': self.get_student_loss()
        }
        
        # 가중치 적용 및 합산
        total_loss = (alpha_feature * loss_components['feature'] + 
                      alpha_distill * loss_components['distillation'] + 
                      alpha_student * loss_components['student'])
        
        return total_loss
    
    
    