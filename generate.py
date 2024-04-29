import requests
import json
from typing import List
from dotenv import load_dotenv, find_dotenv
import prompts
import similarity

def _get_data_through_naver(query: str, category: str, **kwargs) -> json:
    """ Naver 검색 API를 통해 데이터를 수집합니다.
    수집한 데이터는 모델이 generate할 때 RAG로 사용할 수 있습니다.
    Naver 검색 API로 얻는 정보가 부정확할때가 많기 때문에 학습용도로만 이용하면 좋을 것 같습니다.
    자세한 내용은 https://developers.naver.com/docs/serviceapi/search/blog/blog.md를 참고하세요.

    Args:
        query: 검색하려는 쿼리값
        category: 검색하려는 카테고리, 아래 카테고리로 검색이 가능합니다
          + blog: 블로그
          + news: 뉴스
          + kin: 지식인
          + encyc: 백과사전
          + cafearticle: 카페 게시글
          + webkr: 웹문서
        **kwargs: API로 추가적으로 전송할 페이로드

    Returns: resonse

    """

    for k in kwargs:
        if k not in ("display", "start", "sort"):
            raise ValueError("Arguments can be only 'query', 'display', 'start', 'sort'. see: https://developers.naver.com/docs/serviceapi/search/blog/blog.md#%EB%B8%94%EB%A1%9C%EA%B7%B8")

    if category not in ("blog", "news", "kin", "encyc", "cafearticle", "webkr"):
        raise ValueError("category should be one of ('blog', 'news', 'kin', 'encyc', 'cafearticle', 'webkr')")

    _ = load_dotenv(find_dotenv())
    client_id = os.getenv('NAVER_API_ID')
    client_secret = os.getenv('NAVER_API_SECRET')

    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    query = requests.utils.quote(query)
    url = url + f"?query={query}"
    payload = json.dumps(kwargs)

    response = requests.get(url, data=payload, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode())
    else:
        return response.raise_for_status()

def _scoring(query: str, documents: List[str]) -> List:
    """ query와 document의 코사인 유사도를 계산한 뒤, 높은 점수대로 document를 반환하는 함수입니다.

    Args:
        query: 코사인 유사도로 계산할 쿼리
        documents: 코사인 유사도로 계산할 document

    Returns: 코사인 유사도가 높은 순서대로 document 반환

    """

    from transformers import AutoTokenizer, AutoModel

    model_id = "intfloat/e5-base-v2"
    model_device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, device_map=model_device)

    x1 = tokenizer(
        query,
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    x2 = tokenizer(
        documents,
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    x1_output = model(**x1)
    x2_output = model(**x2)

    x1 = similarity.average_pool(model, tokenizer, query)
    x2 = similarity.average_pool(model, tokenizer, documents)

    scores = similarity.cosine_similarity(x1, x2)
    documents, scores = similarity.sort_by_other_iterable(target=documents, key_iter=scores)

    return documents

def generate(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        query: str,
        rag: bool=False,
        **kwargs
) -> str:
    """ 모델의 generate함수입니다.

    Args:
        model: generate할 모델
        tokenizer: 모델의 토크나이저
        query: user가 전송하는 쿼리값
        rag: 네이버 검색 API를 활용한 RAG 사용 여부
        **kwargs: 네이버 검색 API에 전달될 추가 인수

    Returns: 모델이 답변하는 텍스트

    """

    if rag:
        documents = _get_data_through_naver(**kwargs)
        documents = _scoring(query)
        documents = "\n".join(documents[:3])
        query = query +"\n아래 documents를 참조하여 답변하세요\n" + documents

    prompt = prompts.GEMMA_PROMPT.format(question=query)
    inputs = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs)
    completion = tokenizer.decode(
        outputs[0],
        repetition_penalty = 1.5, # Greedy하게 답변하는것을 막기 위함
        temperature = 0.5 # Stochastic한 답변 유도
    )

    return completion

    
