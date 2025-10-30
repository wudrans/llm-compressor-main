##!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   quantize_batch.py
@Time    :   2025/10/29 19:37:25
@Author  :   wlj 
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from utils.file_utils import read_json, get_subfolder, get_exclude_list
from mytools.quantize import myquantize


if __name__ == '__main__':
    DATASET_PATH="/data8T/Text/HuggingFaceH4/ultrachat_200k/data"
    DATASET_SPLIT = "train_sft"
    dataset_path = os.path.join(DATASET_PATH, f"{DATASET_SPLIT}-*.parquet")

    saved_root = "/data8T/models/LLM/Qwen_quantize-awq-sym"

    # get all models
    pretrained_path = "/data/wlj/pretrained/Qwen"
    model_list = get_subfolder(pretrained_path)
    exclude_list = ['Qwen3-30B']
    model_list = get_exclude_list(model_list, exclude_list, exclude_flag=True)
    # model_list = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-30B-A3B-Instruct-2507', 'Qwen3-30B-A3B-Instruct-2507-FP8', 
    #               'Qwen3-30B-A3B-Thinking-2507', 'Qwen3-30B-A3B-Thinking-2507-FP8', 
    #               'Qwen3-4B-Instruct-2507', 'Qwen3-4B-Instruct-2507-FP8', 
    #               'Qwen3-4B-Thinking-2507', 'Qwen3-4B-Thinking-2507-FP8']
    print(model_list)
    
    model_list = ['Qwen3-0.6B']

    for model_name in model_list:
        model_path = os.path.join(pretrained_path, model_name)

        saved_path = os.path.join(saved_root, os.path.basename(model_path))
        os.makedirs(saved_path, exist_ok=True)

        myquantize(model_path, dataset_path, saved_path, dataset_format="parquet")
    