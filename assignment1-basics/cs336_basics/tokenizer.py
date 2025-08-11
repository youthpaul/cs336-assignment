from typing import List, Tuple, Dict, Set
import regex as re
from collections import defaultdict
from collections.abc import Iterable


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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



class bpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer from a given vocabulary, list of merges, and (optionally) special tokens.
        
        Args:
            vocab: A dictionary mapping token IDs to their byte representations.
            merges: A list of tuples representing BPE merge operations.
            special_tokens: Optional list of strings that should be treated as unbreakable tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.bytes2id = {b: i for i, b in vocab.items()}

        # Ensure all the special-tokens in vocab
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.bytes2id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes2id[token_bytes] = new_id
        
        # use for merging, the shortest pair merge first
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text string into a sequence of token IDs.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A list of integer token IDs representing the encoded text.
        """

        id = []

        # Sort special tokens by length (longest first) to avoid partial matches
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if chunk in self.special_tokens:
                id.append(self.bytes2id[chunk.encode('utf-8')])
            else:
                id.extend(self.tokenize_bpe(chunk)) # Otherwise, tokenize normally using BPE

        return id
    

    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        """
        Given an iterable of strings (e.g., a file handle), yield token IDs lazily.
        
        Args:
            iterable: An iterable source of text chunks.
            
        Yields:
            Token IDs generated by processing the input iterable.
        """
        for chunk in iterable:
            yield from self.encode(chunk)
    

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        # Concatenate all token bytes
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        # Decode bytes to string, replacing invalid sequences
        return full_bytes.decode("utf-8", errors="replace")



    def tokenize_bpe(self, text: str) -> list[int]:
        """
        Tokenize a normal piece of text (not a special token) into token IDs.
        
        Args:
            text: A string to tokenize.
            
        Returns:
            A list of token IDs representing the tokenized text.
        """

        # Pre-tokenization
        pre_tokens = []
        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word)

        token_id = []
        for token in pre_tokens:
            # Convert token to bytes tuple
            token_bytes = tuple(bytes([b]) for b in list(token.encode('utf-8')))

            # Apply BPE merges
            merged_token = self.apply_merge(token_bytes)

            # Get token IDs
            token_id.extend(self.bytes2id[b] for b in merged_token)


        return token_id

    
    def apply_merge(self, token_bytes: tuple[bytes, ...]) -> tuple[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            byte_tuple: A tuple of single-byte tokens.
            
        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """

        def get_pairs(token_bytes: list[bytes]):
            """
            get all pairs in token_bytes
            """
            pairs = set()
            prev_char = token_bytes[0]
            for char in token_bytes[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        merge_token = token_bytes
        
        while len(merge_token) > 1:
            pairs = get_pairs(merge_token) # find all pairs
            pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if pair not in self.bpe_ranks:
                break

            first, second = pair # (token1, token2) to be merged
            new_merged_token = []
            i = 0
            while i < len(merge_token):
                try:
                    j = merge_token.index(first, i) # find the first token 'first' behind i
                except ValueError:
                    new_merged_token.extend(merge_token[i:]) # not found
                    break
                else:
                    new_merged_token.extend(merge_token[i:j])
                    i = j

                if merge_token[i] == first and i + 1 < len(merge_token) and \
                    merge_token[i + 1] == second:
                    new_merged_token.append(first + second)
                    i += 2
                else:
                    new_merged_token.append(merge_token[i])
                    i += 1

            # update merge_token
            merge_token = new_merged_token


        return tuple(merge_token)