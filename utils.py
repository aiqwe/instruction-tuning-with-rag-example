import json
import time
import random
import os
from typing import Any, List, Mapping
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

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
def jsave(data:Any, file: str) -> None:
    """ json 파일로 저장합니다.

    Args:
        data: 저장하려는 데이터 객체
        file: 저장 파일명

    Returns: None

    """
    with open(file, "a") as f:
        json.dump(data, f, ensure_ascii=False)

# OpenAI API 함수
def get_completion(prompt: str, model="gpt-3.5-turbo") -> str:
    """ ChatGPT API를 통한 챗봇 기능입니다.
    자세한 내용은 https://github.com/openai/openai-python을 참고하세요.

    Args:
        prompt: ChatGPT에 전송하려는 프롬프트
        model: ChatGPT 모델명

    Returns: ChatGPT의 답변 데이터

    """

    _ = load_dotenv(find_dotenv())
    client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    client.close()

    return response.choices[0].message.content

# Naver에서 크롤링하는 함수
def crawling_by_naver(
        webdriver_path:str = None,
        inputs:List[str] = None,
        save_path:str = None
) -> List[dict]:
    """ Selenium을 통하여 특정 메세지를 검색하고, 결과로 출력되는 네이버 인기글 텍스트 데이터를 수집합니다.

    Args:
        webdriver_path: Selenium에 사용할 Chrome Webriver 위치
        inputs: 검색할 쿼리
        save_path: 수집한 텍스트를 저장할 위치

    Returns: 수집한 데이터

    """

    if not inputs:
        raise ValueError('You should pass "inputs" argument.')

    driver = webdriver.Chrome(webdriver_path)
    url = 'https://www.naver.com'
    driver.get(url)
    search = driver.find_element("id", "query")

    search_data = []
    for question in tqdm(inputs):
        search.send_keys(question)
        search.send_keys(Keys.ENTER)

        # 인기글의 제목
        title_link = driver.find_elements(By.CLASS_NAME, "title_link")
        title = [t.text for t in title_link]
        title = title[:5]

        # 인기글의 텍스트 내용
        dsc_link = driver.find_elements(By.CLASS_NAME, "dsc_link")
        document = [t.text for t in dsc_link]
        document = document[:5]

        search_data.append(dict(question=question, document=document))
        # Blocking을 막기위해 sleep 시간을 랜덤하게 설정
        time.sleep(random.choice(range(20)))
        # 다시 검색하기
        search = driver.find_element("id", "nx_query")
        search.clear()

    if save_path:
        jsave(search_data, save_path)

    return search_data