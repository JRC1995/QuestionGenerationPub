from os import fspath
from pathlib import Path

from transformers_custom import AutoTokenizer, AutoModel

embedding_tag = "t5-base"
save_embedding_path = fspath(Path("../embeddings/T5_base_qg/"))
Path(save_embedding_path).mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(embedding_tag)
model = AutoModel.from_pretrained(embedding_tag)

new_tokens = ["<sep>", "<digit>", "<cls>", "<hl>", "<\hl>"]

all_question_types = ["yesno", "who",
                      "when", "where", "what",
                      "quantity",
                      "how", "why", "which", "other"]

all_question_types = ['<'+qt+'>' for qt in all_question_types]

new_tokens += all_question_types

print(new_tokens)

#print(len(tokenizer))
print(tokenizer.encode("<sep>", add_special_tokens=False))

special_tokens_dict = {'additional_special_tokens': new_tokens}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
#print(len(tokenizer))
print(tokenizer.encode("<sep>", add_special_tokens=False))

model.save_pretrained(save_embedding_path)  # save
tokenizer.save_pretrained(save_embedding_path)  # save