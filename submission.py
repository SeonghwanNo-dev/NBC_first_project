import pandas as pd
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
import Module.classifier as classifier

import Module.TextPreprocessingPipeline as T_Preprocessor
import Module.processed_dataset as processed_dataset

class CustomClassifier(nn.Module):
    """
    트랜스포머 모델(BERT)과 Classification Head를 결합하는 사용자 정의 모델.
    """
    def __init__(self, transformer_model, classification_head):
        super().__init__()
        # 파인튜닝된 트랜스포머 모델 (AutoModel.from_pretrained로 로드된 객체)
        self.transformer = transformer_model 
        # 파인튜닝된 Classification Head (classifier.ClassificationHead 객체)
        self.classifier_head = classification_head

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        # 1. 트랜스포머 모델을 통해 출력(outputs)을 얻음
        # outputs.pooler_output: [CLS] 토큰의 임베딩을 이용한 풀링된 출력 (일반적으로 BERT 분류에 사용됨)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 2. cls vector를 Classification Head에 전달하여 최종 로짓을 계산
        cls_vector = outputs.last_hidden_state[:, 0, :]
        predicted_labels = self.classifier_head(cls_vector)
        
        # 3. Trainer 호환성을 위해 Logits만 튜플로 반환
        return (predicted_labels) # Trainer.predict는 튜플 형태의 출력을 기대합니다.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 모델, 토크나이저 로드
# 저장된 모델 파일 경로 (file_name에 해당)
model = AutoModel.from_pretrained("klue/bert-base").to(DEVICE)
model_path = "/content/drive/MyDrive/Naver_boostCamp/first_project/results/fine_tuning_after_tapt/artifacts/transformer_model.pth"
model_state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(model_state_dict)
model.eval()

classification_head = classifier.ClassificationHead(hidden_size=model.config.hidden_size, num_labels=4).to(DEVICE)
classifier_path = "/content/drive/MyDrive/Naver_boostCamp/first_project/results/fine_tuning_after_tapt/artifacts/classification_head.pth"
classifier_state_dict = torch.load(classifier_path, map_location=DEVICE)
classification_head.load_state_dict(classifier_state_dict)
classification_head.eval()

combined_model = CustomClassifier(
    transformer_model=model,
    classification_head=classification_head
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 2. 데이터로더로 변환
df_test = pd.read_csv("/content/drive/MyDrive/Naver_boostCamp/first_project/upload/data/test.csv")
test_texts = df_test["review"].tolist()
test_data = pd.DataFrame(
    {
        "ID": df_test["ID"],
        "review": test_texts,
        "label": [-1] * len(df_test),  # 테스트 데이터는 레이블 없음 (더미 값)
    }
).reset_index(drop=True)

reviews = test_data['review']

T_Preprocessor_instance = T_Preprocessor.TextPreprocessingPipeline()
# X_test_processed = T_Preprocessor_instance.fit_transform(reviews)

# X_test_processed = reviews.tolist()
X_test_processed = [str(text) for text in reviews.tolist()]

test_full_encodings = tokenizer(
        X_test_processed, 
        truncation=True, 
        padding='max_length', 
        max_length= 256
)

test_dataset = Dataset.from_dict(test_full_encodings)

#2. 추론 시작

# Trainer 초기화
training_args = TrainingArguments(
    output_dir="./tmp_trainer_output_for_inference",
    per_device_eval_batch_size=256,
    do_predict=True,
    report_to="none",
    # 학습 관련 인자는 모두 생략
)

trainer = Trainer(
    model=combined_model,
    args=training_args,
    tokenizer=tokenizer,
)


predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=-1)
df_test["pred"] = predicted_labels
unique_predictions, counts = np.unique(predicted_labels, return_counts=True)


LABEL_MAPPING = {0: "강한 부정", 1: "약한 부정", 2: "약한 긍정", 3: "강한 긍정"}
print("\n클래스별 예측 분포:")
for pred, count in zip(unique_predictions, counts):
    percentage = (count / len(predicted_labels)) * 100
    class_name = LABEL_MAPPING.get(pred, f"클래스 {pred}")
    print(f"   {class_name} ({pred}): {count:,}개 ({percentage:.1f}%)")

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()



# 제출 파일
sample_submission = pd.read_csv("/content/drive/MyDrive/Naver_boostCamp/first_project/upload/data/sample_submission.csv")

submission_df = sample_submission.copy()
submission_df = submission_df[["ID"]].merge(
    df_test[["ID", "pred"]], left_on="ID", right_on="ID", how="left"
)
submission_df = submission_df[["ID", "pred"]]

# 제출 파일 검증
assert len(submission_df) == len(sample_submission), (
    f"길이 불일치: submission_df는 {len(submission_df)}행, sample_submission은 {len(sample_submission)}행"
)
assert submission_df["pred"].isin([0, 1, 2, 3]).all(), (
    "모든 예측값은 [0, 1, 2, 3] 범위에 있어야 합니다"
)
assert not submission_df["pred"].isnull().any(), "예측값에 null 값이 있으면 안됩니다"
assert not submission_df["ID"].isnull().any(), "ID 컬럼에 null 값이 있으면 안됩니다"
print("✅ 모든 검증이 통과되었습니다!")

submission_path = "./output.csv"
submission_df.to_csv(submission_path, index=False)
print(f"제출 파일 저장 완료: {submission_path}")