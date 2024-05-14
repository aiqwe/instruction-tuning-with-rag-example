import torch
from torch.nn import functional as F
from typing import Union, List, Iterable
from transformers import AutoTokenizer, AutoModel
import transformers
from multiprocessing import Pool, cpu_count
from functools import partial

def _infer_device() -> str:
    """ device 자동설정 https://github.com/huggingface/peft/blob/6f41990da482dba96287da64a3c7d3c441e95e23/src/peft/utils/other.py#L75"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif mlu_available:
        return "mlu"
    elif is_xpu_available():
        return "xpu"
    elif is_npu_available():
        return "npu"
    return "cpu"

def _call_default_model(device: str = None):
    """ default 모델 호출 코드를 별도로 작성"""
    if not device:
        device =_infer_device()
    model_id = "intfloat/multilingual-e5-small"
    model = AutoModel.from_pretrained(model_id, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer

def average_pool(
        input_text: Union[List[str], str],
        model: transformers.PreTrainedModel = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        device: str = None
) -> torch.Tensor:
    """Input으로 받는 문장 내 여러 토큰들을 Average Pool을 사용해서 하나의 임베딩 값으로 변환해줍니다..

    Args:
        input_text: 임베딩될 문장 또는 문장의 리스트
        model: 임베딩을 수행할 모델 (default: intfloat/multilingual-e5-base)
        tokenizer: 모델의 토크나이저 (default: intfloat/multilingual-e5-base)
        device: 모델의 device 설정

    Returns: Average Pool된 임베딩 텐서

    """
    if (not model) or (not tokenizer):
        model, tokenizer = _call_default_model(device=device)

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
        input2: Union[List[torch.Tensor], torch.Tensor],
        n_workers: int = cpu_count() - 2
) -> List:
    """input1과 input2의 코사인 유사도를 계산합니다.

    Args:
        input1: 코사인 유사도를 계산할 텐서값
        input2: input1과 코사인 유사도를 계산할 텐서 또는 텐서 리스트
        n_workers: 병렬처리시 작업할 워커의 숫자 (default: cpu_count() - 2)

    Returns: 코사인 유사도 값 리스트

    """
    if isinstance(input2, list):
        with Pool(n_workers) as p:
            scores = p.map(partial(F.cosine_similarity, x1=input1), input2)
    else:
        scores = F.cosine_similarity(x1=input1, x2=input2)

    return scores.tolist()

def sort_by_iterable(target: Iterable, key_iter: Iterable):
    """ key_iter의 내림차순을 기준으로 key_iter, key_iter와 맵핑되는 target을 함께 정렬합니다.
    각각의 document에 대하 유사도 값이 있을 때, 유사도 값을 기준으로 document를 정렬하는데 사용됩니다.

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
        threshold_score: float = None,
        n_workers: int = cpu_count() - 2,
        model: transformers.PreTrainedModel = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
        device: str = None
) -> List:
    """ query와 document의 코사인 유사도를 계산한 뒤, 높은 점수대로 document와 유사도를 반환하는 함수입니다.

    Args:
        query: 코사인 유사도로 계산할 쿼리
        documents: 코사인 유사도로 계산할 document
        threshold_score: 값이 주어지면 해당 값 이상의 값만 필터링
        n_workers: 병렬처리시 작업할 워커의 숫자 (default: cpu_count() - 2)
        model: 유사도 계산시 사용할 모델 (default: multilingual e5 base)
        tokenizer: 유사도 계산시 사용할 모델의 토크나이저
        device: 사용할 디바이스 값 cpu, mps, ... 값이 전달되지 않으면 자동으로 설정

    Returns: 코사인 유사도가 높은 순서대로 document 반환

    """
    if (not model) or (not tokenizer):
        model, tokenizer = _call_default_model(device=device)

    x1 = average_pool(model=model, tokenizer=tokenizer, device=device, input_text=query)
    x2 = average_pool(model=model, tokenizer=tokenizer, device=device, input_text=documents)

    scores = cosine_similarity(input1=x1, input2=x2, n_workers=n_workers)
    documents, scores = sort_by_iterable(target=documents, key_iter=scores)

    if threshold_score:
        documents = [d for d, s in zip(documents, scores) if s >= threshold_score]
        scores = [s for d, s in zip(documents, scores) if s >= threshold_score]

    return documents, scores