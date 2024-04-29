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
    input_text: Union[List[str], str]):
    """Sequence의 Average Pool로 Embedding 값을 계산합니다."""
        
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
    input2: Union[List[torch.Tensor], torch.Tensor]):
    """input1과 input2의 코사인 유사도를 계산합니다."""
    if isinstance(input2, list):
        with Pool(cpu_count()) as p:
            scores = p.map(partial(F.cosine_similarity, x1=input1), input2)
    else:            
        scores = F.cosine_similarity(x1=input1, x2=input2)
        
    return scores.tolist()

def sort_by_iterable(target: Iterable, key_iter: Iterable):
    """ key_iter의 내림차순을 기준으로 key, target을 함께 정렬합니다."""
    pairs = sorted(zip(target, key_iter), key = lambda x: x[1], reverse=True)
    target, key_iter = zip(*pairs)
    target = list(target)
    key_iter = list(key_iter)
    
    return target, key_iter