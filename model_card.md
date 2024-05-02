---

---
## Model Description
**git hub** : [https://github.com/aiqwe/instruction-tuning-with-rag-example](https://github.com/aiqwe/instruction-tuning-with-rag-example)  
github 예제에 따라 gemma-2b-it 모델을 Instruction Tuning한 예제 모델입니다.  
Instruction Tuning에 대해 쉽게 공부할 수 있도록 한글로된 예제 코드를 제공하고 있습니다.

## Usage
### Inference on GPU example
```python
None
```


### Inference on CPU example(Windows)
```python
None
```

### Inference on CPU example(Mac)
```python
None
```


## Chat Template
Gemma 모델의 Chat Template을 사용합니다.  
[gemma-2b-it Chat Template](https://huggingface.co/google/gemma-2b-it#chat-template)

## Training Spec
학습은 구글 코랩 L4 Single GPU를 활용하였습니다.  

| 구분            | 내용                  |
|---------------|---------------------|
| 학습 환경         | Google Colab        |
| GPU           | L4(22.5GB)          |
| 학습시 VRAM      | 약 17GB 사용           |
| dtype         | bfloat16            |
| Attention     | flash attention2    |
| Tuning        | Lora(r=4, alpha=32) |
| Learning Rate | 1e-4                |
| LRScheduler   | Cosine              |
| Optimizer     | adamw_torch_fused   |

## Github
Github : https://github.com/aiqwe
