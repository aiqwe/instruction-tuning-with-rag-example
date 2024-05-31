## Model Description  
**git hub** : [https://github.com/aiqwe/instruction-tuning-with-rag-example](https://github.com/aiqwe/instruction-tuning-with-rag-example)  
Instruction Tuning의 학습을 위해 예시로 학습한 모델입니다.  
[gemma-2b-it](https://huggingface.co/google/gemma-2b-it) 모델을 기반으로 약 1만개의 부동산 관련 Instruction 데이터셋을 학습하였습니다.  
학습 코드는 위 git hub를 참조해주세요.  

## Usage
### Inference on GPU example
```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoPeftModelForCausalLM.from_pretrained(
    "aiqwe/gemma-2b-it-example-v1",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

input_text = "아파트 재건축에 대해 알려줘."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))

```


### Inference on CPU example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoPeftModelForCausalLM.from_pretrained(
    "aiqwe/gemma-2b-it-example-v1",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

input_text = "아파트 재건축에 대해 알려줘."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

### Inference on GPU with embedded function example
내장된 함수로 네이버 검색 API를 통해 RAG를 지원받습니다.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM 
from utils import generate

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoPeftModelForCausalLM.from_pretrained(
    "aiqwe/gemma-2b-it-example-v1",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

rag_config = {
    "api_client_id": userdata.get('NAVER_API_ID'),
    "api_client_secret": userdata.get('NAVER_API_SECRET')
}
completion = generate(
    model=model,
    tokenizer=tokenizer,
    query=query,
    max_new_tokens=512,
    rag=True,
    rag_config=rag_config
)
print(completion)
```

## Chat Template
Gemma 모델의 Chat Template을 사용합니다.  
[gemma-2b-it Chat Template](https://huggingface.co/google/gemma-2b-it#chat-template)
```python
input_text = "아파트 재건축에 대해 알려줘."

input_text = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": input_text}
        ],
        add_generate_prompt=True,
        return_tensors="pt"
    ).to(model.device)

outputs = model.generate(input_text, max_new_tokens=512, repetition_penalty = 1.5)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Training information
학습은 구글 코랩 L4 Single GPU를 활용하였습니다.  

| 구분                          | 내용               |
|-----------------------------|------------------|
| 환경                          | Google Colab     |
| GPU                         | L4(22.5GB)       |
| 사용 VRAM                     | 약 13.8GB         |
| dtype                       | bfloat16         |
| Attention                   | flash attention2 |
| Tuning                      | Lora(r=4, alpha=32) |
| Learning Rate               | 1e-4             |
| LRScheduler                 | Cosine           |
| Optimizer                   | adamw_torch_fused |
| batch_size                  | 4                |
| gradient_accumulation_steps | 2                |