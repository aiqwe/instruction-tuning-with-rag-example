SEED_WORD_PROMPT_PREFIX = """주어진 seed_word에 대해 궁금해할 질문을 10개를 생성하세요.
만들어낸 질문은 JSON형식을 따라야 합니다.
Indentation은 없도록 출력하세요.
아래 양식으로 출력하세요:
"""

SEED_WORD_PROMPT_CONTENT = """{{"seed_word": "{seed_word}", "answer": ["1번째 질문", "2번째 질문"... , "10번째 질문"]}}"""

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

INSTRUCTION_PROMPT_CONTENT = """
###question:
{question}
###document:
{document}
"""

GEMMA_TRAINING_PROMPT = """<bos><start_of_turn>user
아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다.
요청을 적절히 완료하는 응답을 작성해주세요.
{question}<end_of_turn>
<start_of_turn>model
{answer}
"""
