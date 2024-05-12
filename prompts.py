# seed_words.txt에서 질문리스트를 만들때 사용하는 접두어 프롬프트
SEED_WORD_PROMPT_PREFIX = """당신은 부동산에 관심이 많은 사람입니다. 당신의 역할에 따라 주어진 seed_word에 대해 궁금해할 질문을 생성하세요.
seed_word는 총 5개씩 주어집니다.
seed_word에 대해 각각 20개의 질문을 생성해야합니다.
생성할 질문에 대한 요구 사항은 다음과 같습니다:
1. 지시대명사를 사용해서는 안됩니다. seed_word에 있는 명사를 그대로 사용하세요.
2. 이미 만들어낸 질문과 동일하거나 유사한 질문을 만들어내서는 안됩니다.
3. 만들어내는 질문들은 어휘의 다양성을 위해 다양한 단어를 사용해야 합니다.
4. 만들어내는 질문들은 문장의 다양성을 위해 의문문과 평서문을 모두 사용해야 합니다.
5. 반드시 한글로 질문을 만드세요.
6. 만들어낸 질문은 JSON형식을 따라야 하고, indent는 없어야 합니다.
7. 응답하는 답변 문자에는 줄바꿈, \\n, \\t, \\b 등의 특수 문자가 없어야합니다.
8. seed_word에 대해 중복으로 질문을 생성했는지 확인하세요. seed_word에 대한 질문을 이미 생성했다면, 동일한 seed_word에 대한 작업을 해서는 안됩니다.
9. 아래 양식으로 출력하세요:
{{"seed_word": "주어진 seed_word", "answer": ["1번째 질문", "2번째 질문"... , "20번째 질문"]}}

seed_word는 다음과 같습니다:
"""
# seed_words.txt에서 질문리스트를 만들때 사용하는 본문 프롬프트
SEED_WORD_PROMPT_CONTENT = "{seed_word}"

# 카페 게시글 필터링 프롬프트
CAFE_FILTER_PROMPT = """아래 documents들중 다음 규칙을 적용하여 답변하세요.
seed_words와 documents는 리스트 형태로 주어집니다.
요구사항은 다음과 같습니다:
1. 정치적 발언, 혐오 발언 등 독성있고 해로운 documents는 삭제하세요.
2. 리스트 안에 문자열 형태로 주어진 seed_words들과 관련이 있는 데이터만 출력하세요.
3. 출력한 데이터는 JSON형식을 따라야하고, indent는 없어야 합니다.
4. 아래 양식으로 출력하세요.
["document1", "document2", "document3" ...]  

###seed_words:
{seed_words}

###documents:
{documents}

"""


# Training Set을 만들때 사용하는 접두어 프롬프트
INSTRUCTION_PROMPT_PREFIX = """요청받은 question을 document를 참조하여 answer로 답변하세요.
question 1개당 여러개의 document가 주어지며, question은 10개씩 전달됩니다.

요구사항은 다음과 같습니다:
1. 어휘의 다양성을 위해 같은 단어를 반복하지 않습니다.
2. 문장의 형태가 다양해야합니다. 예를 들어 질문과 명령형이 결합되는 형태여야 합니다.
3. 답변은 제공받는 document들을 기반으로 작성되어야 합니다.
4. 제공되는 document의 순서가 먼저 제공될 수록 더 중요한 데이터이므로 답변에 더 많은 영향을 끼쳐야합니다.
5. 답변은 자세한 내용이 포함되도록 제공되어야하지만 200단어를 넘지 않는 것이 좋습니다.

출력 형식은 JSON형식을 따라야 합니다.
Indentation은 없도록 출력하세요.
각 question마다 출력 형식은 다음과 같아야합니다:
{'question': '전달 받은 question의 내용', 'answer': '답변 내용'}
"""

# Training Set을 만들때 사용하는 프롬프트 본문
INSTRUCTION_PROMPT_CONTENT = """
###question:
{question}
###document:
{document}
"""

# Gemma 모델을 학습할 때 사용하는 프롬프트
GEMMA_TRAINING_PROMPT = """<bos><start_of_turn>user
아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다.
요청을 적절히 완료하는 응답을 작성해주세요.
{question}<end_of_turn>
<start_of_turn>model
{answer}
"""