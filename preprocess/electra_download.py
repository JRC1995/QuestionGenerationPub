from os import fspath
from pathlib import Path

from transformers_custom import AutoTokenizer, AutoModel

embedding_tag = "google/electra-base-discriminator"
save_embedding_path = fspath(Path("../../QG_resources/embeddings/ELECTRA_base/"))
Path(save_embedding_path).mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(embedding_tag)
model = AutoModel.from_pretrained(embedding_tag)

"""
new_tokens = ["<sep>", "<digit>", "<cls>"]
#print(len(tokenizer))
print(tokenizer.encode("<sep>", add_special_tokens=False))

special_tokens_dict = {'additional_special_tokens': new_tokens}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
#print(len(tokenizer))
print(tokenizer.encode("<sep>", add_special_tokens=False))
"""

model.save_pretrained(save_embedding_path)  # save
tokenizer.save_pretrained(save_embedding_path)  # save