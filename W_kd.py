from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig

# models = ["naver-hyperclovax/HyperCLOVAX-SEED-Think-14B", "klue/roberta-base", "klue/bert-base", "kykim/bert-kor-base", "beomi/kcbert-base", "monologg/koelectra-base-v3-discriminator"]
# # 1. 가중치 없이 설정 파일만 다운로드하고 로드
# config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
# for i in models:
#     config = AutoConfig.from_pretrained(i, trust_remote_code=True)
#     # 2. 구조 정보 확인
#     # 총 레이어 수 (Transformer 블록 수)
#     num_layers = config.num_hidden_layers
#     # 피처 개수 (Hidden Size, 각 레이어의 출력 차원)
#     hidden_size = config.hidden_size
#     print(f"{i}, Total Layers (num_hidden_layers): {num_layers}")
#     print(f"{i}, Feature Size (hidden_size): {hidden_size}")
    
"""
- Teacher
Total Layers (num_hidden_layers): 38
Feature Size (hidden_size): 6144

- Student
  Total Layers (num_hidden_layers): 12
  Feature Size (hidden_size): 768
"""

layers = [1, 4, 7, 10, 13, 16, 19, 22, 25, 29, 33, 38] # 총 12 레이어


# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(model.config.hidden_size)

# chat = [
#   {"role": "system", "content": "- In this environment, various tools can be used to answer users' questions.\n- You are \"CLOVA X,\" an AI language model developed by NAVER.\n- Begin by creating a plan for solving the problem, and then utilize the tools accordingly to address the problem.\n- The current date is Monday, July 21, 2025.\n- Latest information such as news, stock prices, and shopping is retrieved through the tool_list.\n- If external tools are required, the assistant should not answer directly but must first obtain the necessary information via the assistant -> tool/function_call role, and then respond."},
#   {"role": "user", "content": "Explain in as much detail as possible the relationship between the Schrödinger equation and quantum mechanics."},
# ]

# # By adding skip_reasoning=True, the model is forced to always answer directly without reasoning
# inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, skip_reasoning=True, return_dict=True, return_tensors="pt")
# inputs = inputs.to("cuda")

# output_ids = model.generate(
#     **inputs,
#     max_length=1024,
#     stop_strings=["<|endofturn|>", "<|stop|>"],
#     temperature=0.5,
#     top_p=0.6,
#     repetition_penalty=1.05,
#     tokenizer=tokenizer
# )
# print(tokenizer.batch_decode(output_ids))
