import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import dataclasses
import inspect
import logging
import time

import torch
from utils import load_chat_template, sampling_add_cli_args
from vllm import LLM, EngineArgs, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
)

def run_llm_inference(
    prompts,
    engine_params,
    sampling_params,
    chat_template=None,
    remove_chat_template=False,
):
    """
    大模型推理主函数，供外部直接调用
    参数:
        prompts: List[str]，输入的 prompt 列表
        engine_params: dict，LLM 初始化参数
        sampling_params: SamplingParams 实例
        chat_template: 可选，chat 模型模板
        remove_chat_template: 是否移除 chat 模板
    返回:
        results: List[dict]，每条推理的结果
        num_tokens: int，总 token 数
        qps: float，QPS
    """
    logging.info(f"开始推理，prompts 数量: {len(prompts)}")
    model_name = os.path.dirname(engine_params["model"]).rsplit("/")[-1]
    logging.info(f"加载模型: {model_name}")
    llm = LLM(**engine_params)

    # chat template
    if remove_chat_template:
        if "chat" in model_name.lower():
            logging.warning(
                f"The model name from model path is {model_name}, so we guess you are using the chat model and the additional processing is required for the input prompt. "
                f"If the result is not quite correct, please ensure you do not pass --remove_chat_template in CLI."
            )
        prompts_new = prompts
    else:
        logging.info("正在处理 chat 模型模板...")
        logging.warning(
            "If you are using a non chat model, please pass the --remove_chat_template in CLI."
        )
        try:
            load_chat_template(llm.get_tokenizer(), chat_template)
            prompts_new = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = llm.get_tokenizer().apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts_new.append(text)
            logging.info("chat 模型模板处理完成。")
        except Exception as e:
            logging.warning(
                f"use tokenizer apply_chat_template function failed, may because of low transformers version...(try use transformers>=4.34.0), error: {e}"
            )
            prompts_new = prompts

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    logging.info("开始生成...")
    outputs = (
        llm.generate(prompts_new, sampling_params)
        if isinstance(prompts_new[0], str)
        else llm.generate(sampling_params=sampling_params, prompt_token_ids=prompts_new)
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time
    logging.info(f"生成结束，耗时: {duration_time:.2f} 秒")

    num_tokens = 0
    results = []
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        num_tokens += token_count
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "token_count": token_count
        })
        logging.info(f"Prompt: {prompt}\nGenerated text: {generated_text}\nToken count: {token_count}")

    qps = num_tokens / duration_time if duration_time > 0 else 0
    logging.info(f"总 tokens: {num_tokens}, QPS: {qps:.2f}")
    return results, num_tokens, qps

def build_engine_and_sampling_params(args):
    engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
    sampling_args = [
        param.name
        for param in list(
            inspect.signature(SamplingParams).parameters.values()
        )
    ]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    sampling_params_dict = {
        attr: getattr(args, attr) for attr in sampling_args if args.__contains__(attr)
    }
    sampling_params = SamplingParams(**sampling_params_dict)
    return engine_params, sampling_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument(
        "--remove_chat_template",
        default=False,
        action="store_true",
        help="pass this if you are not use a chat model",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser = sampling_add_cli_args(parser)
    args = parser.parse_args()

    engine_params, sampling_params = build_engine_and_sampling_params(args)

    # 示例 prompt
    prompts = [
        "哪些迹象可能表明一个人正在经历焦虑?"
    ]

    results, num_tokens, qps = run_llm_inference(
        prompts,
        engine_params,
        sampling_params,
        chat_template=args.chat_template,
        remove_chat_template=args.remove_chat_template,
    )

    for item in results:
        print(f"Prompt: {item['prompt']}\nGenerated text: {item['generated_text']}\n")
    print(f"tokens: {num_tokens}, QPS: {qps}")