import json
import time
import random
import os
import requests
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
            return list(f)
        if ext == "json":
            return json.load(f)

# JSON파일로 저장하기
def jsave(data:Any, file: str, mode:Literal["a", "w"] = "a", indent: int=None) -> None:
    """ json 파일로 저장합니다.

    Args:
        data: 저장하려는 데이터 객체
        file: 저장 파일명

    Returns: None

    """
    with open(file, mode) as f:
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

    # 브라우저없이 크롤링
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    url = 'https://www.naver.com'
    driver.get(url)
    search = driver.find_element("id", "query")

    search_data = []
    for question in tqdm(inputs):
        search.send_keys(question)
        search.send_keys(Keys.ENTER)

        # 인기글의 텍스트 내용
        dsc_link = driver.find_elements(By.CLASS_NAME, "dsc_link")
        document = [t.text for t in dsc_link]
        document = document[:n_documents]

        search_data.append(dict(question=question, document=document))
        # Blocking을 막기위해 sleep 시간을 랜덤하게 설정
        time.sleep(random.choice(range(20)))
        # 다시 검색하기
        search = driver.find_element("id", "nx_query")
        search.clear()

    if save_path:
        jsave(search_data, save_path, indent=indent)

    return search_data

def generate(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        query: str,
        repetition_penalty: float = 1.5,
        temperature: float = 0.5,
        max_new_tokens = 256,
        rag: bool=True,
        n_documents: int = 5
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
        n_documents: RAG 사용시 검색 및 참조할 인기글의 갯수

    Returns: 모델이 답변하는 텍스트

    """

    if rag:
        search_data = get_document_through_selenium(inputs=query, n_documents=n_documents, save_path=False)
        documents = list(map(lambda x: x['document'], search_data))
        documents, _ = similarity.sort_by_similarity(query, documents)
        documents = "\n".join(documents)
        query = query +(
            "\n아래 documents를 참조하여 답변하세요. 먼저 제공되는 documents를 더 많이 참조하세요\n"
            "documents:\n"
        ) + documents

    prompt = prompts.GEMMA_PROMPT.format(question=query)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    completion = tokenizer.decode(outputs[0])

    return completion