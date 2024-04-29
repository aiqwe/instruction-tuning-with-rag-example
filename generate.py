import requests
import json
from typing import List
from dotenv import load_dotenv, find_dotenv
import prompts
import similarity

def _get_naver_data(query: str, category: str, **kwargs):
    
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
    
def _scoring(query: str, documents: List):
    
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
    rag: bool=False):
    
    if rag:
        documents = _scoring(query)
        documents = "\n".join(documents[:3])
        query = query +"\n아래 documents를 참조하여 답변하세요\n" + documents
    
    prompt = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=150)
    completion = tokenizer.decode(
        outputs[0],
        repetition_penalty = 1.5,
        temperature = 0.5
    )
    
    return completion

    
