from transformers import AutoTokenizer
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset

import Module.experiment_tool as exp_tool
import Module.setSeed as setSeed
import Module.TextPreprocessingPipeline as T_Preprocessor
import Module.tapt as tapt

RANDOM_STATE = 42
setSeed.set_seed(RANDOM_STATE)

# tool 사용하기
PATH_TO_STORE = './results'
PROJECT_NAME = 'klue_bert-base_tapt_v1'
HW_COUNT = torch.cuda.device_count()
HW_NAME = torch.cuda.get_device_name(0)
CONFIG = {
    "seed": RANDOM_STATE,
    "lr": 1e-5, 
    "batch_size": 16, 
    "num_epochs": 3,
    "model": "klue/bert-base",
    "device_name": HW_NAME,
    "device_count": HW_COUNT,
}
e_tool = exp_tool.ExperimentTool(PATH_TO_STORE, PROJECT_NAME, CONFIG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = "klue/bert-base"

# 데이터 로드
base_dir = Path(__file__).resolve().parent
train_file_path = base_dir/'data/train.csv'
test_file_path = base_dir/'data/test.csv'
df_train = pd.read_csv(train_file_path, encoding='utf-8')
df_test = pd.read_csv(test_file_path, encoding='utf-8')
df_combined = pd.concat([df_train['review'], df_test['review']], ignore_index=True)

# 전처리
T_Preprocessor_instance = T_Preprocessor.TextPreprocessingPipeline()
text_processed = T_Preprocessor_instance.fit_transform(df_combined)

# dataset 저장
e_tool.d_log(text_processed, "combined_text_processed")
text_processed = text_processed.tolist()

# Main Function
# --- TAPT 학습 데이터셋 준비 (Hugging Face datasets 사용) ---
# 전처리된 텍스트 리스트를 DataFrame으로 변환 후 Dataset 객체 생성
df_tapt = pd.DataFrame({'text': text_processed})
tapt_dataset = Dataset.from_pandas(df_tapt)
tokenizer = AutoTokenizer.from_pretrained(model)

# 토크나이징 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=256)

# 데이터셋 토크나이징 실행
tokenized_dataset = tapt_dataset.map(
    tokenize_function, 
    batched=True,
    num_proc=4, # 병렬 처리로 속도 향상
    remove_columns=["text"]
)
tapt_instance = tapt.tapt(model_name=model, tokenizer=tokenizer, e_tool=e_tool)
# tapt_instance.train(dataset=tokenized_dataset, epoch_num=3)
resume_from_checkpoint_path = f"{e_tool.artifact_path}/checkpoint-37000"
tapt_instance.train(dataset=tokenized_dataset, resume_from_checkpoint=resume_from_checkpoint_path, epoch_num=3)
tapt_instance.save("tapted_teacher")