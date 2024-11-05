import pathlib
import time
import argparse
import os.path
from model import Huggingface_Models
from utils import *
import datetime
from tqdm import tqdm
from wandb.sdk.data_types.trace_tree import Trace
import google.generativeai as genai
import wandb
from openai import OpenAI
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

model_path = {
    "blip2": "blip2-opt-2.7b",
    "instructblip": "instructblip-vicuna-13b",
    "internlm": "internlm-xcomposer2-vl-7b",
    "mplug": "mplug-owl-llama-7b",
    "bakllava": "bakLlava-v1-hf",
    "llava_mistral": "llava-v1.6-mistral-7b-hf",
    "llava_vicuna": "llava-1.5-13b-hf",
    "deepseek": "deepseek-vl-7b-chat",
    "minicpm": "MiniCPM-Llama3-V-2_5",
    "qwen": "Qwen-VL",
}


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
def run_model_vqa(model, model_name, img_path, prompt, ground_truth=None, max_new_token=20):
    token_usage = {}
    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    try:
        if "gpt" in model_name.lower():
            if model_name == 'gpt-4v':
                model_name = 'gpt-4-turbo'
            api_key = os.environ.get("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": "Output your choice (option name, e.g., A, B, etc.) first."
                        }]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_new_token,
                temperature=0.0,
            )
            response = res.choices[0].message.content

        elif 'claude' in model_name:
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            if model_name == 'claude_opus':
                model_name = 'claude-3-opus-20240229'
            elif model_name == 'claude_sonnet':
                model_name = 'claude-3-sonnet-20240229'
            message = client.messages.create(
                model=model_name,
                max_tokens=max_new_token,
                temperature=0.0,
                system="Output your choice (option name, e.g., A, B, etc.) first.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(img_path),
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            response = message.content[0].text

        elif 'gemini' in model_name:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-pro')
            cookie_picture = {
                'mime_type': 'image/jpeg',
                'data': pathlib.Path(img_path).read_bytes()
            }
            response = model.generate_content(["System Instruction: output your choice (option name, e.g., A, B, etc.) first.\n"+prompt, cookie_picture],
                                              generation_config=genai.types.GenerationConfig(
                                                      # Only one candidate for now.
                                                      candidate_count=1,
                                                      max_output_tokens=max_new_token,
                                                      temperature=0))
            time.sleep(5.0)
            response = response.text

        else:
            response = model.vqa(img_path, prompt, max_new_token)
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "success"
        status_message = (None,)
        response_text = response
        if response_text[0] == '(':
            response_text = response_text[1:]
    except Exception as e:
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        status = "error"
        status_message = str(e)
        response_text = " "
    root_span = Trace(
        name="root_span",
        kind="llm",
        status_code=status,
        status_message=status_message,
        metadata={
            "token_usage": token_usage,
            "model_name": model_name,
        },
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"query": prompt},
        outputs={"response": response_text,
                 "ground_truth": ground_truth},
    )
    root_span.log(name="llm_trace")
    return response_text


def evaluate(args, name):
    if args.model in ['blip2', 'instructblip', 'bakllava', 'llava_mistral', 'internlm', 'mplug', 'llava_vicuna',
                      'minicpm', 'qwen']:
        model = Huggingface_Models(args.model, model_path[args.model], args.device)
    else:
        model = None

    eval_data = sampling_test_data(args.cache_dir)

    acc = 0
    res = []
    graph_group_acc, task_group_acc = defaultdict(int), defaultdict(int)
    binary_acc, mcq_acc = 0, 0
    binary_count = 0
    for item_i, item in tqdm(enumerate(eval_data)):
        item['prompt'] = ''
        options = item['options']
        answer_index = item['answer_index']
        option_text = convert_options(options)
        item['prompt'] += f"Question: {item['question']}\nChoose from the following options:\n{option_text}\n"
        if args.model in ['bakllava', 'llava_mistral', 'llava_vicuna']:
            item['prompt'] = prompt_answer_with_input(item['prompt'], "mcq", args.model)
        else:
            item['prompt'] += "The best answer is: ("

        prompt = item['prompt']
        img = get_image_dir(item['img_id'])
        ground_truth = chr(ord('A') + answer_index)
        response_text = run_model_vqa(model, args.model, img, prompt, ground_truth, 30)
        res.append({
            "data_id": item_i,
            "img_id": item["img_id"],
            "prompt": prompt,
            "options": options,
            "response": response_text,
            "graph_type": item["graph_type"],
            "task_type": item["task_type"],
            "graph": item["graph"],
            "objs": item["objs"],
            "ground_truth": ground_truth
        })
        if response_text[0] == ground_truth:
            graph_group_acc[item['graph_type']] += 1
            task_group_acc[item['task_type']] += 1
            if len(options) == 2:
                binary_acc += 1
            else:
                mcq_acc += 1
            acc += 1
        if len(options) == 2:
            binary_count += 1
    save_output(os.path.join(args.output_dir, args.dataset, args.model), res, f'{name}.json')
    print("Overall Accuracy is: %.02f\n" % (acc / len(eval_data)))
    wandb.log({'Accuracy': acc / len(eval_data)})

    graph_type_counts = count_specific_types(eval_data, "graph_type")
    print(graph_type_counts)
    for graph_type in graph_type_counts:
        wandb.log({f'{graph_type} Accuracy': graph_group_acc[graph_type] / graph_type_counts[graph_type]})

    task_type_counts = count_specific_types(eval_data, "task_type")
    print(task_type_counts)
    for task_type in task_type_counts:
        wandb.log({f'{task_type} Accuracy': task_group_acc[task_type] / task_type_counts[task_type]})

    wandb.log({'Binary Accuracy': binary_acc / binary_count})
    wandb.log({'MCQ Accuracy': mcq_acc / (len(eval_data) - binary_count) if len(eval_data) != binary_count else 0})
    print(binary_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default='./data/cello_data.jsonl', type=str)
    parser.add_argument("--output_dir", default='./output', type=str)
    parser.add_argument("--dataset", default='CELLO', type=str)
    parser.add_argument("--model", default='gpt-4o', type=str)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    set_seed()
    name = args.dataset + '_' + args.model
    wandb.init(project="CELLO", config=args, name=name)
    evaluate(args, name)


if __name__ == "__main__":
    main()
