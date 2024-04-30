## GPT With Tuning
ChatGPT / Claude3를 이용하여 데이터셋을 만들어 Instruction Tuning을 수행한 예제입니다.  
학습과 추론 단계에서 네이버 검색 API를 활용하여 조금이라도 더 모델이 정확한 답변을 줄 수 있도록 예제를 만들었습니다.  
간단한 코드 트리는 아래와 같습니다.  

![process](assets/process.svg)  

[preprocess.ipynb](preprocess.ipynb)에서 데이터를 전처리하고, ChatGPT와 네이버 검색 API를 통해 Instruction 데이터셋을 생성합니다.  
[train.ipynb](train.ipynb)에서 훈련 데이터셋을 PEFT로 학습시켜 모델을 튜닝합니다.  
튜닝이 끝난 후 huggingface에 있는 모델 예제를 통해 간단하게 추론해볼 수 있습니다.  

---

### References
이 자료는 아래 Reference들을 참조하였습니다.  
학습이 목적이라면 아래 자료들을 꼭 확인해주세요.  
+ [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
+ [Prompt Engineering(deeplearning.ai)](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers)
+ [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
+ [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
+ [Fine-Tuning Gemma Models in Hugging Face](https://huggingface.co/blog/gemma-peft)
+ [Openai API - Python](https://github.com/openai/openai-python)
+ [Flash Attention](https://github.com/Dao-AILab/flash-attention)
+ [네이버 API 가이드](https://developers.naver.com/docs/common/openapiguide/)  

예제 모델은 아래 링크를 참조해주세요.  
+ [https://huggingface.co/aiqwe/gemma-2b-it-sgtuned](https://huggingface.co/aiqwe/gemma-2b-it-sgtuned)

---

### 각 파일들의 역할은 다음과 같습니다.
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
