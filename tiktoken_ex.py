import tiktoken

"""
Note:
    There is always a trade off between vocalbullary size and sequence lenght of integers
    (vocab_size inversly proptional to sequence length)
    if vocab_size == small
        sequence_len will be large
    if vocab_size == large
        sequence_len will be short

    In practice: people use sub word tokenizer
"""

enc = tiktoken.get_encoding('gpt2')

print(f"vocalbulary size={enc.n_vocab}")

encode = enc.encode('hi there')

print(f'example of encoding the text hi there = {encode}')

decode = enc.decode(encode)

print(f' the decoded text of hi there = {decode}')