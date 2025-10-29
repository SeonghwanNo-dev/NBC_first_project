import torch.nn as nn

class HintProjector(nn.Module):
    """
    Teacher와 Student 모델의 히든 스테이트 차원이 다를 때,
    차원을 일치시키기 위한 선형 변환(Projection) 레이어입니다.
    """
    def __init__(self, t_dim: int, s_dim: int, middle_dim:int):
        super().__init__()
        # MLP 헤드 정의 (흔히 ReLU와 Dropout을 포함)
        self.projector = nn.Sequential(
            nn.Linear(s_dim, middle_dim), # D -> 중간 차원
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(middle_dim, t_dim)
        )

    def forward(self, transformer_output):
        
        # 프로젝터에 통과시켜 최종 로짓(logits) 얻기
        projected_output = self.projector(transformer_output)
        return projected_output