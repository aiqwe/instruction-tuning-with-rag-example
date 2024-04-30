import torch
from torch.nn import functional as F
from typing import Union, List, Iterable
from transformers import AutoTokenizer, AutoModel
import transformers
from multiprocessing import Pool, cpu_count
from functools import partial

def average_pool(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        input_text: Union[List[str], str]
) -> torch.Tensor:
    """Sequence의 Average Pool로 Embedding 값을 계산합니다.

    Args:
        model: 엠베딩을 수행할 인코더 모델
        tokenizer: 모델의 토크나이저
        input_text: Sequence

    Returns: 임베딩 텐서

    """

    inputs = tokenizer(
        input_text,
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    output = model(**inputs)
    last_hidden = output.last_hidden_state.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0)
    embeddings = (last_hidden.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]).to("cpu")
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def cosine_similarity(
        input1: torch.Tensor,
        input2: Union[List[torch.Tensor], torch.Tensor]
) -> List:
    """input1과 input2의 코사인 유사도를 계산합니다.

    Args:
        input1: 코사인 유사도를 계산할 텐서값
        input2: input1과 코사인 유사도를 계산할 텐서 또는 텐서 리스트

    Returns: 코사인 유사도 리스트 객체

    """
    if isinstance(input2, list):
        with Pool(cpu_count()) as p:
            scores = p.map(partial(F.cosine_similarity, x1=input1), input2)
    else:
        scores = F.cosine_similarity(x1=input1, x2=input2)

    return scores.tolist()

def sort_by_iterable(target: Iterable, key_iter: Iterable):
    """ key_iter의 내림차순을 기준으로 key, target을 함께 정렬합니다.

    Args:
        target: 정렬하려는 타겟 이터러블 객체
        key_iter: 정렬의 기준이 되는 이터러벌 객체

    Returns: key_iter의 오름차순에 따라 정렬된 target, key_iter

    """
    pairs = sorted(zip(target, key_iter), key = lambda x: x[1], reverse=True)
    target, key_iter = zip(*pairs)
    target = list(target)
    key_iter = list(key_iter)

    return target, key_iter

def sort_by_similarity(
        query: str,
        documents: List[str],
        device: str = "cpu"
) -> List:
    """ query와 document의 코사인 유사도를 계산한 뒤, 높은 점수대로 document를 반환하는 함수입니다.

    Args:
        query: 코사인 유사도로 계산할 쿼리
        documents: 코사인 유사도로 계산할 document

    Returns: 코사인 유사도가 높은 순서대로 document 반환

    """

    from transformers import AutoTokenizer, AutoModel

    model_id = "intfloat/e5-base-v2"
    model_device = device
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, device_map=model_device)

    x1 = average_pool(model, tokenizer, query)
    x2 = average_pool(model, tokenizer, documents)

    scores = cosine_similarity(x1, x2)
    documents, scores = sort_by_iterable(target=documents, key_iter=scores)

    return documents