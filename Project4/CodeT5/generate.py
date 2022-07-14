from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('codet5')
model = T5ForConditionalGeneration.from_pretrained('codet5')

text = """def getCurrentOrigin(self):
    return self.prevKs[-1]"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length=64)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
