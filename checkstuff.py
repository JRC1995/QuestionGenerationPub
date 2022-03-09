import torch

from models.transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch as T
import torch.nn.functional as F

model_name = "ahotrod/electra_large_discriminator_squad2_512"

"""
# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = [{
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}, {'question': 'Why am I so stupid?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'}]
_, res = nlp(QA_input, handle_impossible_answer=True)

print(res)
"""

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
questions = ['Why is model conversion important?', "Define Model"]
contexts = [
    'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.',
    'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.']

batch_input_ids = []
batch_token_type = []
batch_attn_mask = []
for question, context in zip(questions, contexts):
    input_dict = tokenizer.encode_plus(text=question, text_pair=context, add_special_tokens=True)
    batch_input_ids.append(input_dict["input_ids"])
    batch_attn_mask.append(input_dict["attention_mask"])
    if "token_type_ids" in input_dict:
        batch_token_type.append((input_dict["token_type_ids"]))
    else:
        batch_token_type = None

max_len = max([len(sample) for sample in batch_input_ids])

for i in range(len(batch_input_ids)):
    sample = batch_input_ids[i]
    if len(sample) < max_len:
        sample = sample + [tokenizer.pad_token_id] * (max_len - len(sample))
        batch_input_ids[i] = sample
        sample = batch_attn_mask[i]
        sample = sample + [0] * (max_len - len(sample))
        batch_attn_mask[i] = sample
        if batch_token_type is not None:
            sample = batch_token_type[i]
            sample = sample + [0] * (max_len - len(sample))
            batch_token_type[i] = sample

if batch_token_type is not None:
    batch_token_type = T.tensor(batch_token_type).cuda()
batch_input_ids = T.tensor(batch_input_ids).cuda()
batch_attn_mask = T.tensor(batch_attn_mask).cuda()

print(batch_attn_mask.size())

with torch.no_grad():
    out_dict = model(input_ids=batch_input_ids,
                     attention_mask=batch_attn_mask,
                     token_type_ids=batch_token_type,
                     return_dict=False)
    start_logits = out_dict[0]
    end_logits = out_dict[1]

    start_logits = start_logits * batch_attn_mask + (1 - batch_attn_mask) * -100000
    end_logits = end_logits * batch_attn_mask + (1 - batch_attn_mask) * -100000

    N, S = start_logits.size()

    question_mask = T.cat([T.ones(N, 1).float().to(batch_token_type.device),
                           batch_token_type[:, 1:].float()], dim=1)

    start_logits = start_logits * question_mask + (1 - question_mask) * -100000
    end_logits = end_logits * question_mask + (1 - question_mask) * -100000

    start_logits = F.softmax(start_logits, dim=-1)
    end_logits = F.softmax(end_logits, dim=-1)

    # start_logits = start_logits * batch_attn_mask + (1-batch_attn_mask) * -100000
    # end_logits =

    print("start_logits: ", start_logits)
    print("end_logits: ", end_logits)
    print("impossibility score: ", start_logits[:, 0] * end_logits[:, 0])
