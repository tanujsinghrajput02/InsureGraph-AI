
import os
import time
import torch

from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv(
    "HF_LLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct"
)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def generate_answer(model, tokenizer, inputs):

    device = next(model.parameters()).device

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print("Model parameter dtype:", next(model.parameters()).dtype)

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    elapsed = time.perf_counter() - start

    generated_tokens = outputs[0][input_ids.shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip(), elapsed


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": """
You are an insurance assistant.

Policy information:
Customer: Rahul Sharma
Policy Number: POL-1001
Coverage Amount: INR 500000

Question:
What is the coverage amount for Rahul Sharma?

Give a short answer.
"""
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    print("\nLoading original FP32 model...")

    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    original_model = original_model.float()
    original_model.eval()

    print(
        "Original model dtype:",
        next(original_model.parameters()).dtype
    )

    print("\nTesting original model...")

    original_answer, original_time = generate_answer(
        original_model,
        tokenizer,
        inputs
    )

    print("Original answer:", original_answer)
    print("Original generation time:", round(original_time, 2), "seconds")

    print("\nPreparing INT8 quantization...")

    supported_engines = torch.backends.quantized.supported_engines
    print("Supported quantization engines:", supported_engines)

    if "onednn" in supported_engines:
        torch.backends.quantized.engine = "onednn"
    elif "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "x86" in supported_engines:
        torch.backends.quantized.engine = "x86"
    else:
        print("No supported quantization engine found.")
        return

    quantization_start = time.perf_counter()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        original_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    quantization_time = time.perf_counter() - quantization_start

    quantized_model.eval()

    print("Quantization time:", round(quantization_time, 2), "seconds")

    print("\nTesting INT8 model...")

    quantized_answer, quantized_time = generate_answer(
        quantized_model,
        tokenizer,
        inputs
    )

    print("INT8 answer:", quantized_answer)
    print("INT8 generation time:", round(quantized_time, 2), "seconds")

    improvement = (
        (original_time - quantized_time) / original_time
    ) * 100 if original_time > 0 else 0

    print("\n========== BENCHMARK RESULTS ==========")
    print("Original time:", round(original_time, 2), "seconds")
    print("INT8 time:", round(quantized_time, 2), "seconds")
    print("Quantization setup:", round(quantization_time, 2), "seconds")
    print("Speed improvement:", round(improvement, 2), "%")
    print("Original answer:", original_answer)
    print("INT8 answer:", quantized_answer)
    print("=======================================")


if __name__ == "__main__":
    main()
