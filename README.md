# 어떤 도움을 줄 수 있을까요?
외국어로된 Instruction Tuning 예제는 많았지만, 한글로된 예제는 찾기가 어렵습니다.  
특히나 End-to-end 예제는 영문으로된 예제도 찾기 어려웠습니다.(대부분 train.py만 만들어 놓고 데이터는 이미 가공된 데이터를 사용)  

ChatGPT는 어떻게 학습하지? 라는 것이 궁금하신 LLM 입문자이시면, 본 예제가 작은 도움이 되지 않을까하여 작성해보았습니다.

# 목차
[개요](#개요)  
[훈련 스펙](#훈련-스펙)  
[Colab 실행 가이드](#Colab-실행-가이드)  
[코드 트리](#코드-트리)  
[참조 문서](#참조-문서)  


# 개요
본 예제는  
1. [부동산에 관련된 학습 데이터셋](data/instruction.jsonl)을 만들어보고
2. [Gemma모델](https://huggingface.co/google/gemma-2b-it)에 학습시켜
직접 생성형 모델을 만들어보는 것이 목적입니다.

![process](assets/process.svg)  

[preprocess.ipynb](preprocess.ipynb)  
데이터를 전처리하는 과정을 담았습니다. 좋은 데이터셋을 만들기 위해 Claude Web 버젼과 GPT4 API를 사용하였습니다.
  
[train.ipynb](train.ipynb)  
훈련 데이터셋을 PEFT로 학습시켜 모델을 튜닝합니다. 데이터셋을 `SFTTrainer`에 전달하기 위해 데이터셋을 마스킹하고 배치작업을 위한 패딩, `Dataset` 생성 등 학습을 위한 작업코드들이 포함되어 있습니다.

# 훈련 스펙
학습은 쉽게 실험해 볼 수 있도록 [Google Colab](https://colab.google/)을 사용했으며, 훈련시 확인된 스펙은 아래와 같았습니다.

|구분|내용|
|-|-|
|환경|Google Colab|
|GPU|L4(22.5GB)|
|학습시 VRAM|약 17GB 사용|
|dtype|bfloat16|
|Attention|flash attention2|
|Tuning|Lora(r=4, alpha=32)|
|Learning Rate|1e-4|
|LRScheduler|Cosine|
|Optimizer|adamw_torch_fused|

Colab에서 A100은 자주 연결이 끊어지기 때문에 안정적인 L4 GPU로 훈련하였습니다.  

# Colab 실행 가이드


# 코드 트리
각 파일들은 아래의 역할을 수행합니다.

|구분|파일명| 역할                                     |
|-|-|----------------------------------------|
|노트북|[preprocess.ipynb](preprocess.ipynb)| 데이터셋을 만드는 노트북 예제                       |
|노트북|[train.ipynb](train.ipynb)| 학습 코드 예제                               |
|데이터|[data/seed_words.txt](data/seed_words.txt)| 학습하려는 도메인의 키워드 모음                      |
|데이터|[data/query.jsonl](data/query.jsonl)| 키워드기반으로 생성한 질문리스트                      |
|데이터|[data/search_data.json](data/search_data.json)| 질문리스트로 검색한 네이버 인기글 모음                  |
|데이터|[data/instruction.jsonl](data/instruction.jsonl)| 검색데이터와 질문리스트로 생성한 Instruction 데이터셋     |
|모듈|[utils.py](utils.py)| json읽기, OpenAI API, generate등 각종 도움 기능 |
|모듈|[similarity.py](similarity.py)| RAG에서 사용할 데이터를 랭킹 작업해주는 기능 모음          |
|모듈|[prompts.py](prompts.py)| 프롬프트 모음                                |

# 참조 문서
본 예제는 아래 문서들을 참조하였습니다.
+ [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
+ [Prompt Engineering(deeplearning.ai)](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers)
+ [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
+ [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
+ [Fine-Tuning Gemma Models in Hugging Face](https://huggingface.co/blog/gemma-peft)
+ [Openai API - Python](https://github.com/openai/openai-python)
+ [Flash Attention](https://github.com/Dao-AILab/flash-attention)
+ [네이버 API 가이드](https://developers.naver.com/docs/common/openapiguide/)  

예제의 모델은 아래 링크를 참조해주세요.  
+ [https://huggingface.co/aiqwe/gemma-2b-it-sgtuned](https://huggingface.co/aiqwe/gemma-2b-it-sgtuned)
