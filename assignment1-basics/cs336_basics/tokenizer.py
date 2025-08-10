from typing import List, Tuple, Dict, Set
import regex as re
from collections import defaultdict

def merge_token(s: Tuple, best_pair: Tuple, new_token: bytes):
    # merge all pair (token1, token2) to new_token
    res = []
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i : i + 2] == best_pair:
            res.append(new_token)
            i += 2
        else:
            res.append(s[i])
            i += 1
    return tuple(res)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # initialization
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    cur_vocab_idx = 256 # the next index of new merged token
    merges: List[Tuple[bytes, bytes]] = []

    # add special_tokens to vocab
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[cur_vocab_idx] = token_bytes
            cur_vocab_idx = cur_vocab_idx + 1
    
    # load text data
    text = ""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # split the text by <|endoftext|>
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)

    # pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_cnt = defaultdict(int) # the count of token
    bpe_cnt = defaultdict(int) # the count of bytes pair: (byte1, byte2)
    for chunk in chunks:
        for token in re.finditer(PAT, chunk):
            token = token.group(0)
            token_bytes = token.encode('utf-8')
            bytes_list = [bytes([x]) for x in token_bytes] # e.g. ['h', 'e', 'l']
            token_cnt[tuple(bytes_list)] += 1
    
    # compute bpe_cnt
    for bytes_list, cnt in token_cnt.items():
        for i in range(len(bytes_list) - 1):
            bpe_cnt[bytes_list[i], bytes_list[i + 1]] += cnt
    
    # train the BPE
    while len(vocab) < vocab_size:
        if not bpe_cnt:
            break

        max_cnt = max(bpe_cnt.values()) # the max frequency
        candidates = [k for k, v in bpe_cnt.items() if v == max_cnt] # find all most frequently pairs
        best_pair = max(candidates) # lexicographically biggest
        merges.append(best_pair) # record this merge

        # add new token to vocab
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[cur_vocab_idx] = new_token_bytes
        cur_vocab_idx += 1

        # find all pairs that need to update
        affected_token = []
        for token, cnt in token_cnt.items():
            tag = any(token[i : i + 2] == best_pair for i in range(len(token) - 1))
            if tag:
                affected_token.append((token, cnt))
        
        # update the bpe_cnt
        for token, cnt in affected_token:
            # substract old frequenct
            for i in range(len(token) - 1):
                bpe_cnt[token[i], token[i + 1]] -= cnt
                if bpe_cnt[token[i], token[i + 1]] <= 0:
                    del bpe_cnt[token[i], token[i + 1]]
            
            # merge a pair
            new_token = merge_token(token, best_pair, new_token_bytes)

            # update new frequency
            for i in range(len(new_token) - 1):
                bpe_cnt[new_token[i], new_token[i + 1]] += cnt
            del token_cnt[token]
            token_cnt[new_token] += cnt
            
            # print()
            # print(token)
            # print(new_token)
            # break
        
        # print('\n=================\n')
        # print(bpe_cnt[(b' t', b't')]) # 0?
        # print(bpe_cnt[(b' ', b'a')])
        # print('\n=================\n')
        # break




    return vocab, merges
