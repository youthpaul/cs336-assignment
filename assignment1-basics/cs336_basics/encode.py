import pickle
import json
import os
import pathlib
import sys
from tokenizer import bpeTokenizer
import numpy as np
from tqdm import tqdm

import multiprocessing as mp
from token_utils import find_chunk_boundaries
import regex as re

import time
import logging
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

special_tokens = ["<|endoftext|>"]

def process_chunk(
    file_path: str,
    start_byte: int,
    end_byte: int,
    tokenizer,
) -> np.ndarray:
    """处理单个文本分块的核心函数"""
    chunk_size = end_byte - start_byte
    
    # 只读取指定字节范围 (避免加载整个文件)
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        chunk_data = f.read(chunk_size)

    # Sort special tokens by length (longest first) to avoid partial matches
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_special_tokens))

    # 将字节数据解码为字符串
    text = chunk_data.decode('utf-8', errors='ignore')   # 忽略解码错误，以防万一

    return tokenizer.encode(text)

def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):

    print(f'cpu count: {mp.cpu_count()}')

    num_chunks = mp.cpu_count() - 1

    # Find chunk boundaries
    chunk_boundaries = find_chunk_boundaries(
        path=path_to_txt,
        num_desired_chunks=num_chunks,
        special_split_tokens=[bytes(token.encode('utf-8')) for token in special_tokens], 
    )

    # Number of chunks to process in parallel
    # it can be less than the number of chunks
    num_processes = len(chunk_boundaries) - 1

    # create tasks list to processed
    tasks = [
        (
            path_to_txt,                
            chunk_boundaries[i],         
            chunk_boundaries[i + 1],     
            tokenizer,                   
        )
        for i in range(num_processes)
    ]
    
    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)

    # Use a multiprocessing pool to process the chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Map the token_cnt function to the tasks
        results = pool.starmap(process_chunk, tasks)

        # merge
        results = [id for l in results for id in l]

        # 存储为 .npy 文件（二进制格式）
        np.save(save_path, results)
    
    # data = np.memmap(save_path, dtype=np.int32, mode='r')
    # print(data)



if __name__ == "__main__":
    
    # 读取词表和merges
    with open('data/tinystories_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('data/tinystories_merges.pkl', 'rb') as f:
        merges = pickle.load(f)

    # 构造tokenizer
    tokenizer = bpeTokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens
    )

    # Track memory and time
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    encode_txt_as_numpy_array(tokenizer, 'data/TinyStoriesV2-GPT4-train.txt', 
                              'data/tinystories_train_tokens.npy')
    encode_txt_as_numpy_array(tokenizer, 'data/TinyStoriesV2-GPT4-valid.txt', 
                              'data/tinystories_val_tokens.npy')

    # Calculate elapsed time and memory usage
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    elapsed_time = end_time - start_time

    # Display results
    hrs = elapsed_time // 3600
    mins = (elapsed_time % 3600) // 60
    secs = elapsed_time % 60
    
    logging.info(f"Encoding complete in {hrs:.0f}h {mins:.0f}m {secs:.1f}s")
    logging.info(f"Memory usage: {end_memory - start_memory:.2f} MB")