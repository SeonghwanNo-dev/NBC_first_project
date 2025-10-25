'''
Argument: path_to_store, project_name, config
Method
- log: 학습 과정에서 변화하는 Loss, Accuracy 같은 값들을 기록
- d_log: dataset을 경로에 기록
- w_log: weight를 경로에 기록
'''
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import pandas as pd


class ExperimentTool:
    """
    WandB를 대체하여 딥러닝 실험의 설정, 로그, 아티팩트(데이터셋/가중치)를 
    로컬 파일 시스템에 체계적으로 저장하고 관리하는 도구입니다.

    이는 실험의 재현성 확보와 결과 비교를 용이하게 합니다.
    """

    def __init__(self, path_to_store: str, project_name: str, config: Dict[str, Any]):
        """
        ExperimentTool을 초기화합니다. 저장 경로를 설정하고, 
        실험 설정을 저장하며, 필요한 디렉토리를 생성합니다.

        Args:
            path_to_store (str): 모든 결과 파일을 저장할 기본 디렉토리 경로. 
            project_name (str): 현재 실험이 속한 프로젝트 이름.
            config (Dict[str, Any]): 하이퍼파라미터 및 고정된 실험 설정 딕셔너리.
        """
        # 기본 저장 경로 설정 및 생성 (Pathlib 사용)
        self.project_name = project_name
        self.base_path = Path(path_to_store)/self.project_name
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.config = config

        # 로그 파일을 저장할 경로 설정 (CSV 사용)
        self.log_file = self.base_path / "run_log.csv"

        # 모델 가중치 및 데이터셋을 저장할 아티팩트 디렉토리 생성
        self.artifact_path = self.base_path / "artifacts"
        self.artifact_path.mkdir(exist_ok=True)

        # 설정 파일 저장
        self._save_config()

    def _save_config(self) -> None:
        """
        현재 실험 설정을 JSON 파일로 저장합니다.
        """
        config_path = self.base_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            # 설정은 YAML 또는 JSON 형태로 저장하는 것이 일반적입니다.
            json.dump(self.config, f, indent=4)
        print(f"설정 파일 저장 완료: {config_path}")

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        학습 과정에서 변화하는 Loss, Accuracy 같은 스칼라 지표를 기록합니다. 
        파일이 존재하지 않으면 헤더를 먼저 생성합니다.

        Args:
            metrics (Dict[str, Any]): epoch, loss, acc 등의 지표를 담은 딕셔너리.
                예시: {'epoch': 1, 'train_loss': 0.5, 'test_acc': 0.85}
        """
        log_data = {
            "epoch": metrics.get("epoch"),              # 현재 에포크 번호
            "train_loss": metrics.get("train_loss"),    # 훈련 데이터의 평균 손실
            "test_acc": metrics.get("test_acc"),        # 검증/테스트 데이터의 정확도
            "learning_rate": self.config.get("lr"),     # 학습률 (하이퍼파라미터)
            "batch_size": self.config.get("batch_size"),# 배치 크기 (하이퍼파라미터)
        }
        
        # 파일 존재 여부 확인 (헤더 작성 필요 여부 판단)
        file_exists = self.log_file.exists()

        # CSV 파일에 데이터 행 추가
        # mode='a' (append)를 사용하지만, 파일이 없으면 생성됨
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            fieldnames = log_data.keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # [수정] 파일이 존재하지 않았다면, 헤더를 먼저 작성합니다.
            if not file_exists:
                writer.writeheader() 
                
            writer.writerow(log_data)
    
    def d_log(self, dataset: pd.DataFrame, name: str) -> str:
        """
        실험에 사용된 데이터셋을 CSV 형태로 저장합니다.

        Args:
            dataset (pd.DataFrame): 저장할 데이터셋 (Pandas DataFrame).
            name (str): 데이터셋의 이름 (예: 'train_data_final').

        Returns:
            str: 저장된 파일의 절대 경로.
        """
        data_path = self.artifact_path / f"{name}.csv"
        dataset.to_csv(data_path, index=False)
        print(f"데이터셋 저장 완료: {data_path}")
        return str(data_path)

    def w_log(self, model: nn.Module, name: str) -> str:
        """
        학습된 모델의 가중치(Weights)를 PyTorch .pth 파일로 저장합니다.

        Args:
            model (nn.Module): 저장할 PyTorch 모델 객체.
            name (str): 모델 가중치의 이름 (예: 'best_model_weights').

        Returns:
            str: 저장된 파일의 절대 경로.
        """
        weight_path = self.artifact_path / f"{name}.pth"
        torch.save(model.state_dict(), weight_path)
        print(f"모델 가중치 저장 완료: {weight_path}")
        return str(weight_path)


# --------------------------------------------------------------------------------------------------
# [사용 예시]
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 테스트를 위한 더미 데이터 및 모델 설정
    PATH_TO_STORE = './NBC_first_project/results'
    PROJECT_NAME = 'test_v1'
    CONFIG = {
        "lr": 2e-5, 
        "batch_size": 16, 
        "num_epochs": 10,
        "model_name": "klue/roberta-base"
    }

    # 1. ExperimentTool 초기화
    tool = ExperimentTool(PATH_TO_STORE, PROJECT_NAME, CONFIG)

    # 2. log 테스트 (학습 과정 기록)
    print("\n--- 학습 과정 기록 테스트 ---")
    for epoch in range(1, 4):
        # 가상의 학습 결과
        train_loss = 0.5 - epoch * 0.05
        test_acc = 0.70 + epoch * 0.05
        
        tool.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "test_acc": test_acc
        })
        print(f"Epoch {epoch} 지표 기록 완료.")
    
    # 3. d_log 테스트 (데이터셋 저장)
    print("\n--- 데이터셋 저장 테스트 ---")
    dummy_data = pd.DataFrame({
        'review': ['a', 'b', 'c'], 
        'label': [1, 0, 1]
    })
    data_path = tool.d_log(dummy_data, name="initial_dataset_version")

    # 4. w_log 테스트 (모델 가중치 저장)
    print("\n--- 모델 가중치 저장 테스트 ---")
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)
            
    model = DummyModel()
    model_path = tool.w_log(model, name="roberta_best_head_epoch_3")
    
    print(f"\n모든 기록은 폴더 {tool.base_path}에 저장되었습니다.")
