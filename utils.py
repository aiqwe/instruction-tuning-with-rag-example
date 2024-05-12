import json
import time
import random
import os
import requests
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Any, List, Mapping, Literal, Union
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import transformers
import similarity
import prompts


def _getenv(key):
    """ 환경변수를 찾는 함수입니다. .env """
    load_dotenv(find_dotenv())
    return os.getenv(key)

def jload(file: str) -> List | str:
    """ json 또는 json line 파일을 로드합니다.

    Args:
        file: json 또는 json line 파일명

    Returns: json일 경우 json 객체를 반환, json line일 경우 리스트 객체 반환

    """
    with open(file, "r") as f:
        ext = file.split(".")[-1]
        if ext == "jsonl":
            data = list(f)
            result = []
            for idx in range(len(data)):
                try:
                    decoded = json.loads(data[idx])
                    result.append(decoded)
                except:
                    raise ValueError(
                        f"Can't decode to json at index [{idx}].\n"
                        "You should check whether json format is correct or not first."
                    )
            return result
        if ext == "json":
            return json.load(f)

# JSON파일로 저장하기
def jsave(data:Union[List[Any], Any], save_path: str, mode:Literal["a", "w"] = "a", indent: int=None) -> None:
    """ json 파일로 저장합니다.

    Args:
        data: 저장하려는 데이터 객체
        file: 저장 파일명

    Returns: None

    """
    with open(save_path, mode) as f:
        if save_path.endswith("jsonl"):
            if indent:
                raise ValueError("If you are trying to save in 'jsonl' format, you must not assign 'indent' argument.")
            for d in data:
                json.dump(obj=d, fp=f, ensure_ascii=False, indent=indent)
                f.write("\n")
        else:
            json.dump(obj=data, fp=f, ensure_ascii=False, indent=indent)

# OpenAI API 함수
def get_completion(prompt: str, model="gpt-3.5-turbo", api_key: str = None) -> str:
    """ ChatGPT API를 통한 챗봇 기능입니다.
    자세한 내용은 https://github.com/openai/openai-python을 참고하세요.

    Args:
        prompt: ChatGPT에 전송하려는 프롬프트
        model: ChatGPT 모델명

    Returns: ChatGPT response중 답변 텍스트 데이터

    """

    if not api_key:
        api_key = _getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI api key is not setted.")

    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    client.close()

    return response.choices[0].message.content

def get_document_through_selenium(
        inputs:Union[List[str], str] = None,
        n_documents: int = 7,
        save_path:Union[bool, str] = None,
        indent: int = 4
) -> List[dict]:
    """ Selenium을 통하여 특정 메세지를 검색하고, 결과로 출력되는 네이버 인기글 텍스트 데이터를 수집합니다.

    Args:
        inputs: 검색할 쿼리
        n_documents: 수집할 인기글 데이터 갯수
        save_path: 수집한 텍스트를 저장할 위치
        indent: json 저장시 indentation값

    Returns: [{'question': 검색어, 'title': 인기글 제목, 'document': 인기글 요약 내용}]의 리스트 형태로 수집 데이터를 저장

    """

    if not inputs:
        raise ValueError('You should pass "inputs" argument.')
    if isinstance(inputs, str):
        inputs = [inputs]

    with Pool(cpu_count()) as pool:
        search_data = list(tqdm(pool.imap(partial(_selenium_crawling, n_documents=n_documents), inputs)))

    if save_path:
        jsave(search_data, save_path, indent=indent)

    return search_data

def _selenium_crawling(question, n_documents):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    url = 'https://www.naver.com'
    driver.get(url)
    search = driver.find_element("id", "query")

    search.send_keys(question)
    search.send_keys(Keys.ENTER)

    # 인기글의 텍스트 내용
    dsc_link = driver.find_elements(By.CLASS_NAME, "dsc_link")
    document = [t.text for t in dsc_link]
    document = document[:n_documents]
    driver.close()

    return dict(question=question, document=document)

def get_document_through_api(
        query: str,
        api_client_id: str = None,
        api_client_secret: str = None,
        category: Literal["blog", "news", "kin", "encyc", "cafearticle", "webkr"] = "news",
        display: int = 20,
        start: int = 1,
        sort: str = "sim"
) -> json:
    """ Naver 검색 API를 통해 데이터를 수집합니다.
    수집한 데이터는 모델이 generate할 때 RAG로 사용할 수 있습니다.
    Naver 검색 API로 얻는 정보가 부정확할때가 많기 때문에 학습용도로만 이용하면 좋을 것 같습니다.
    자세한 내용은 https://developers.naver.com/docs/serviceapi/search/blog/blog.md를 참고하세요.

    Args:
        query: 검색하려는 쿼리값
        api_client_id: 네이버 검색 API이용을 위한 발급 받은 client_id 값
          - 환경변수는 'NAVER_API_ID'로 설정하세요
        api_client_secret: 네이버 검색 API이용을 위한 발급 받은 client_secret 값
          - 환경변수는 'NAVER_API_SECRET'으로 설정하세요
        category: 검색하려는 카테고리, 아래 카테고리로 검색이 가능합니다
          - blog: 블로그
          - news: 뉴스
          - kin: 지식인
          - encyc: 백과사전
          - cafearticle: 카페 게시글
          - webkr: 웹문서
        display: 검색 결과 수 지정, default = 20
        start: 검색 페이지 값
        sort: 정렬값
          - 'sim': 정확도 순으로 내림차순 정렬
          - 'date': 날짜 순으로 내림차순 정렬

    Returns: API로부터 제공받은 검색 결과 response값

    """

    if not (api_client_id and api_client_secret):
        api_client_id = _getenv("NAVER_API_ID")
        api_client_secret = _getenv("NAVER_API_SECRET")
    if not api_client_id or not api_client_secret:
        id_ok = "'NAVER_API_ID'" if not api_client_id else ""
        secret_ok = "'NAVER_API_SECRET'" if not api_client_id else ""
        raise ValueError(f"{id_ok} {secret_ok} Not setted")

    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": api_client_id, "X-Naver-Client-Secret": api_client_secret}

    query = requests.utils.quote(query)
    url = url + f"?query={query}"
    payload = dict(display=display, start=start, sort=sort)
    payload = json.dumps(payload)

    response = requests.get(url, data=payload, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode())
    else:
        return response.raise_for_status()

def generate(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        query: str,
        repetition_penalty: float = 1.5,
        temperature: float = 0.5,
        max_new_tokens = 256,
        rag: bool=False,
        rag_config:Mapping = None
) -> str:
    """ 모델의 generate함수입니다.

    Args:
        model: generate할 모델
        tokenizer: 모델의 토크나이저
        query: user가 전송하는 쿼리값
        repetition_penalty:
        temperature:
        max_new_tokens:
        rag: 네이버 검색 API를 활용한 RAG 사용 여부
        rag_config: 네이버 검색 API에 전달될 추가 설정값
          - api_client_id: 네이버 검색 API의 CLIENT_ID
          - api_client_secret: 네이버 검색 API의 SECRET
          - score: 검색 결과를 필터링할 유사도 점수 기준값

    Returns: 모델이 답변하는 텍스트

    """

    if (rag and not rag_config):
        raise ValueError(
            "If you want to use RAG, pass 'rag_config' with 'api_client_id' and 'api_client_secret'"
            "example)"
            "rag_config={'api_client_id'}: 'your_client_id', 'api_client_scret': 'your_client_secret'"
        )
    if (rag and rag_config):
        categories = ["blog", "news", "kin", "encyc", "cafearticle", "webkr"]
        documents = []
        for category in categories:
            search_docs = get_document_through_api(query, category=category, **rag_config)
            search_docs = [content['description'] for content in search_docs['items']]
            search_docs = [docs.replace("<b>", "").replace("</b>", "") for docs in search_docs]
            if not len(search_docs) == 0:
                search_docs, scores = similarity.sort_by_similarity(query, search_docs)
                search_docs = [d for d, s in zip(search_docs, scores) if s >= 0.9]  # Cosine Similarity가 0.9 이상인 문서만 수집
            documents = documents + search_docs
        query = query + (
            "\nIf you need more information, search information through below <documents> tag."
            "\nYou should answer with Korean."
            "\n<documents>"
            "\n-"
        ) + "\n -".join(documents) if len(documents) != 0 else query

    inputs = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": query}
        ],
        add_generate_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return completion