import base64
import torch
import random
import numpy as np
import os
import json
from collections import defaultdict
import io


def get_image_dir(img_id):
    dict_dir1 = './Visual_Genome/VG_100K/'
    dict_dir2 = './Visual_Genome/VG_100K_2/'
    if os.path.exists(dict_dir1+str(img_id)+'.jpg'):
        return dict_dir1+str(img_id)+'.jpg'
    elif os.path.exists(dict_dir2+str(img_id)+'.jpg'):
        return dict_dir2+str(img_id)+'.jpg'

def encode_image(image_path):
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            return image_path
        else:
            with open(image_path, "rb") as image_file:
                b64 = base64.b64encode(image_file.read()).decode('utf-8')
            return f"{b64}"
    else:
        buffered = io.BytesIO()
        image_path.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"{b64}"

def sampling_test_data(pth, sample_size=100):
    data = []
    with open(pth, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    task_type_dict = defaultdict(list)
    for item in data:
        task_type_dict[item['task_type']].append(item)

    sampled_data = []

    for task_type, items in task_type_dict.items():
        if len(items) < sample_size:
            sampled_data.extend(items)
        else:
            sampled_data.extend(random.sample(items, sample_size))

    return sampled_data

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def convert_options(options):
    formatted_options = []

    for i, option in enumerate(options):
        formatted_option = f"({chr(65 + i)}). {option}"
        formatted_options.append(formatted_option)

    result = '\n'.join(formatted_options)
    return result

B_INST_LLAMA, E_INST_LLAMA = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt_llama = f"{B_SYS}You are a helpful chat assistant and will answer the user's questions carefully.{E_SYS}"

phrase_answer_multiple_choice = "The best answer is: ("
phrase_answer_open_ended = "The best short answer is:"

is_chat_model = True  # TODO: so far for all models used here

def prompt_answer(c_task, E_INST):
    if c_task == 'open_ended':
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_open_ended}\n"""
    else:
        return f"""{E_INST if is_chat_model else ''}{phrase_answer_multiple_choice}"""


def prompt_answer_with_input(inputt, c_task, model_name="bakllava"):
    if "bakllava" == model_name:
        B_INST_IMG, B_INST, E_INST = "USER: <image>\n", "USER:\n", "\nASSISTANT:\n"
    elif "llava_mistral" == model_name:
        B_INST_IMG, B_INST, E_INST = "[INST]: <image>\n", "[INST] ", " [/INST] "
    elif "llava_vicuna" == model_name:
        system_prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
        B_INST_IMG, B_INST, E_INST = f"{system_prompt} USER: <image>\n", "USER:\n", "\nASSISTANT:\n"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented yet.")

    return f"""{B_INST_IMG if is_chat_model else ''}{inputt}{prompt_answer(c_task, E_INST)}"""

def save_output(path, res, file_name='res.json'):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), 'w') as f:
        json.dump(res, f)
    f.close()

def count_specific_types(data, key_name):
    graph_type_counts = {}
    for item in data:
        graph_type = item.get(key_name)
        if graph_type in graph_type_counts:
            graph_type_counts[graph_type] += 1
        else:
            graph_type_counts[graph_type] = 1

    return graph_type_counts