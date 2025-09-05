from agentturing.constants import LLM_MODEL
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch, os


def get_llm():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Inside get_llm")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("loading tokenizer")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
    print("loading model")
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        return_full_text=False,
        pad_token_id=tok.eos_token_id,
    )

    return pipe, tok