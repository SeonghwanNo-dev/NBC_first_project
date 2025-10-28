from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

class tapt:
    def __init__(self, model_name, tokenizer, e_tool):
        # 4. 마스크 언어 모델링(MLM) 데이터 콜레이터 설정
        # TAPT의 목표인 MLM 학습을 위해 동적으로 마스크를 생성합니다.
        self.tokenizer = tokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.e_tool = e_tool
        self.trainer = 0 # train 메서드에서 할당

    def train(self, dataset, epoch_num):
        data_collator = DataCollatorForLanguageModeling(
        tokenizer=self.tokenizer, 
        mlm=True, 
        mlm_probability=0.15 # 마스킹 비율 (일반적으로 15%)
        )
        # 5. 학습 인자(Arguments) 설정
        # 학습률은 기존 사전 학습보다 낮게 설정하는 경우가 많습니다.
        training_args = TrainingArguments(
            output_dir=self.e_tool.artifact_path,
            num_train_epochs=epoch_num,  # TAPT 에폭 (적절한 값 설정 필요)
            per_device_train_batch_size=16,
            save_steps=3000,                  # 체크포인트 저장 간격
            save_total_limit=2,               # 저장할 최대 체크포인트 수
            logging_steps=50,                 # 로그 기록 주기
            prediction_loss_only=True,        # 예측 손실만 계산 (MLM은 주로 손실만 봄)
            learning_rate=1e-5,               # 미세 조정(Fine-tuning)보다 낮은 학습률 사용
            logging_dir=self.e_tool.log_file,
        )
        
        self.trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        )
        self.trainer.train()
        
    # 최종 학습된 모델의 가중치만 저장
    def save(self, file_name):
        self.trainer.save_model(file_name)