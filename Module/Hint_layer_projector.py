import torch.nn as nn
import torch

class HintProjector(nn.Module):
    """
    Teacher와 Student 모델의 히든 스테이트 차원이 다를 때,
    차원을 일치시키기 위한 선형 변환(Projection) 레이어입니다.
    """
    def __init__(self, t_dim: int, s_dim: int):
        """
        Args:
            t_dim (int): Teacher Model의 Hidden State 차원.
            s_dim (int): Student Model의 Hidden State 차원.
        """
        super().__init__()
        # Teacher의 특징 차원을 Student의 특징 차원에 맞춥니다.
        # Student Feature <- Teacher Feature @ W_h
        self.projection = nn.Linear(t_dim, s_dim)

    def forward(self, t_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_feature (torch.Tensor): Teacher Model의 Hint Layer 출력 (batch_size, seq_len, t_dim).
        Returns:
            torch.Tensor: 차원이 조정된 특징 벡터 (batch_size, seq_len, s_dim).
        """
        return self.projection(t_feature)