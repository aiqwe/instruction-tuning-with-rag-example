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

# 카페 게시글 필터링 접두어 프롬프트
CAFE_PROMPT_PREFIX = """당신은 부동산에 관심이 많은 사람입니다. documents를 요구사항에 따라 필터링하고 질문을 생성해야합니다.
documents는 20개가 주어지며, 필터링한 documents를 수정하여 명확하고 구어적인 표현으로 질문을 생성해야 합니다.
요구사항은 다음과 같습니다:
1. 정치적 발언, 혐오 발언 등 독성있고 해로운 documents는 제외하세요.
2. 부동산 분야와 관련된 documents만 높은 기준으로 필터링하세요.
3. 필터링한 documents로 사람들이 궁금해 할만한 질문을 생성하세요.
4. 필터링한 documents는 출력 결과에 포함하지마세요.
5. 출력한 데이터는 JSON형식을 따라야하고, indent는 없어야 합니다.
6. 아래 양식으로 출력하세요.
생성한 질문1
생성한 질문2
...

예시1:
documents = '청약시 무주택자격 질문'
생성한 질문 = '청약시 무주택 자격이 어떻게 되나요?'

예시2:
documents = '여의도 매수하려고 하는데요.'
생성한 질문 = '여의도 매수 관하여 질문드립니다.'

예시3:
documents = '서울 아파트 매물이 빠른 속도로 소진되고 있네요...'
생성한 질문 = '서울 아파트 매물이 추세가 어떻게 되고 있나요?'

예시4:
documents = '매매 전세 동시진행'
생성한 질문 = '매매와 전세를 동시에 진행하는데 주의할 점이 뭘까요?'

예시5:
documents = '3기신도시 당해 아닌경우'
생성한 질문 = '3기신도시 당해 조건에 해당하지 않는 경우 문의드립니다.'
"""

# 카페 게시글 필터링 본문 프롬프트
CAFE_PROMPT_CONTENT = """
###documents:
{documents}
"""


# Training Set을 만들때 사용하는 접두어 프롬프트
INSTRUCTION_PROMPT_PREFIX = """요청받은 question을 document를 참조하여 answer로 답변하세요.
question 1개당 여러 개의 document가 주어지며, question은 각각 10개씩 전달됩니다.

요구사항은 다음과 같습니다:
1. 어휘의 다양성을 위해 같은 단어를 반복하지 않습니다.
2. 문장의 형태가 다양해야합니다. 예를 들어 질문과 명령형이 결합되는 형태여야 합니다.
3. document의 정보가 question과 관련있는 정보라면 참조합니다, 관련이 없다면 참조하지 않습니다.
4. WebPilot을 이용하여 Google에서 검색을 하고, question과 관련있는 데이터를 참조하세요.
5. 위 3, 4번의 데이터를 바탕으로 답변하세요.
6. 답변은 자세한 내용이 포함되도록 제공되어야하지만 200단어 정도로 구성해주세요.
9. 출력 형식은 JSON포맷을 따라야 하며, Indentation은 없도록 출력하세요.

출력 형식은 아래 포맷을 참조하세요:
{"question": "1주택자 종부세 폐지 여부가 확정된 건가요?", "answer": "현재 1주택자에 대한 종합부동산세 폐지...."}
{"question": "아파트 추이 분석에서 중요한 변수들은 무엇인가요?", "answer": "아파트 추이를 이해하기 위해 ...."}"""

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

EVAL_BATTLE_PROMPT = """
[System]
공정한 심사위원이 되어 2개의 AI 모델이 제공하는 답변의 품질을 평가하세요.
사용자의 지시를 잘 따르고, 사용자의 질문에 더 잘 답변하는 Assistant를 선택하세요.
답변의 유용성, 관련성, 정확성, 깊이, 창의성 및 세부 수준과 같은 요소를 고려하여 평가해야 합니다.
가장 중요하게 고려할 사항은 정확성입니다.
최대한 객관적으로 판단해야 하며, 어떠한 입장의 bias도 있어선 안됩니다.
답변이 제시된 순서가 판단에 영향을 미치지 않도록 해야합니다.
답변의 길이가 평가에 영향을 미치지 않도록 합니다.
특정 Assistant의 이름을 선호해서는 안됩니다.
가능한 객관적으로 평가해야합니다.

평가 후 최종 평점을 산출해야 합니다:
- A Assistant의 답변이 더 좋으면 "A", B Assistant의 답변이 더 좋으면 "B", 동점이면 "C"로 표기합니다.

평가후 JSON 포맷에따라 평점과 설명을 출력하세요.
Markdown 양식은 제거하고 출력하세요.
출력할 JSON 스키마는 아래와 같습니다.
{{"selected_assistant": selected_assistant: str, "reason": reason: str}}
- selected_assistant : 선택한 Assistant
- reason : selected_assistant를 선택한 이유

[사용자 질문]
{question}
[사용자 질문]
[Assistant A의 답변 시작]
{answer_a}
[Assistant A의 답변 종료]
[Assistant B의 답변 시작]
{answer_b}
[Assistant B의 답변 종료]
"""

EVAL_SCORE_PROMPT="""
[System]
공정한 심사위원이 되어 AI 모델이 제공하는 답변의 품질을 평가하세요.
답변의 유용성, 관련성, 정확성, 깊이, 창의성 및 세부 수준과 같은 요소를 고려하여 평가해야 합니다.
가장 중요하게 고려할 사항은 정확성입니다.
최대한 객관적으로 판단해야 하며, 어떠한 입장의 bias도 있어선 안됩니다.
답변이 제시된 순서가 판단에 영향을 미치지 않도록 해야합니다.
답변의 길이가 평가에 영향을 미치지 않도록 합니다.
특정 Assistant의 이름을 선호해서는 안됩니다.

평가후 JSON 포맷에따라 평점과 설명을 출력하세요.
Markdown 양식은 제거하고 출력하세요.
출력할 JSON 스키마는 아래와 같습니다.
{{"score": score: int, "reason": reason}}
- score : 평가 점수
- reason : score 점수 측정 사유

[사용자 질문]
{question}
[Assistant의 답변 시작]
{answer}
[Assistant의 답변 종료]
"""