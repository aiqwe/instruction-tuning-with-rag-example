import json
import time
import random
import os
from typing import Any, List
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# JSON / JSON LINE 파일 불러오기
def jload(file: str):
    with open(file, "r") as f:
        ext = file.split(".")[-1]
        if ext == "jsonl":
            return list(f)
        if ext == "json":
            return json.load(f)

# JSON파일로 저장하기
def jsave(data:Any, file: str):
    with open(file, "a") as f:
        json.dump(data, f, ensure_ascii=False)
    
# OpenAI API 함수
def get_completion(prompt, model="gpt-3.5-turbo"):
    
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
def crawling_by_naver(webdriver_path:str = None, inputs:List[str] = None, save_path:str = None):

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