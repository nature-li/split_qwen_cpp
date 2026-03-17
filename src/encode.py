from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("/workspace/models/Qwen2.5-3B-Instruct")

# encode
messages = [{"role": "user", "content": "讲一下 transformer"}]
text = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = t.encode(text)
print("Token IDs:", ids)
print("Length:", len(ids))
