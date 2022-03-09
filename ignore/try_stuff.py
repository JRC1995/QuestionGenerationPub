#[32099, 13, 3, 2020, 12, 3, 19, 1791, 33, 3, 384, 19, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("../preprocess/embeddings/T5_base/")

print(tokenizer.decode([3]))