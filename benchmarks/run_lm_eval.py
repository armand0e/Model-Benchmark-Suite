import argparse
import torch
from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM


def run_lm_eval(
    model_name,
    tasks_list=["gpqa_diamond_zeroshot"],
    limit=None,
    batch_size=1,
    quantization="4bit",
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.1,
    hf_token=None,
    backend="hf",
):
    print(
        f"Running LM Eval on {model_name} with tasks {tasks_list}, "
        f"{quantization} quantization, backend={backend}..."
    )

    # Authenticate with HuggingFace if a token is provided (for private/gated models)
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        print("Authenticated with HuggingFace token.")

    if backend == "vllm":
        # vLLM backend: faster inference, Linux/WSL only
        model_args = f"pretrained={model_name},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True"
        if hf_token:
            model_args += f",token={hf_token}"

        results = evaluator.simple_evaluate(
            model="vllm",
            model_args=model_args,
            tasks=tasks_list,
            num_fewshot=0,
            batch_size="auto",
            limit=limit,
            gen_kwargs={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": True,
            },
            log_samples=True,
        )
    else:
        # HuggingFace Transformers backend (default)
        if quantization == "4bit":
            model_args = (
                f"pretrained={model_name},load_in_4bit=True,trust_remote_code=True"
            )
        elif quantization == "8bit":
            model_args = (
                f"pretrained={model_name},load_in_8bit=True,trust_remote_code=True"
            )
        else:  # None or bf16
            model_args = (
                f"pretrained={model_name},dtype=bfloat16,trust_remote_code=True"
            )

        # Pass token in model_args so the model loader can access private repos
        if hf_token:
            model_args += f",token={hf_token}"

        # Auto-detect device instead of hard-coding CUDA
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
            print("CUDA not available, using CPU. This may be slow for large models.")

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks_list,
            num_fewshot=0,
            batch_size=batch_size,
            limit=limit,
            device=device,
            gen_kwargs={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": True,
            },
            log_samples=True,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        default="gpqa_diamond_zeroshot",
        help="Comma separated list of tasks",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",")]
    results = run_lm_eval(args.model, task_list, args.limit)
    print(results)
