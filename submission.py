import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
from torch.utils.data import DataLoader

import Module.TextPreprocessingPipeline as T_Preprocessor
import Module.processed_dataset as processed_dataset

# 1. 모델, 토크나이저 로드
# 저장된 모델 파일 경로 (file_name에 해당)
model_path = "./results/klue_bert-base_tapt_v1/artifacts/tapted_teacher"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 데이터로더로 변환
df_test = pd.read_csv("./data/test.csv")
test_texts = df_test["review"].tolist()
test_data_1 = pd.DataFrame(
    {
        "ID": df_test["ID"],
        "review": test_texts,
        "label": [-1] * len(df_test),  # 테스트 데이터는 레이블 없음 (더미 값)
    }
).reset_index(drop=True)

reviews = df_test['review']
labels = df_test['label']

T_Preprocessor_instance = T_Preprocessor.TextPreprocessingPipeline()
X_test_processed, y_test = T_Preprocessor_instance.fit_transform(reviews, labels)

X_test_processed = X_test_processed.tolist()
y_test = y_test.tolist()

test_full_encodings = tokenizer(
        X_test_processed, 
        truncation=True, 
        padding='max_length', 
        max_length= 256
)

test_dataset = processed_dataset.processed_dataset(test_full_encodings, y_test)

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
    model=model,
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
sample_submission = pd.read_csv("data/sample_submission.csv")

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