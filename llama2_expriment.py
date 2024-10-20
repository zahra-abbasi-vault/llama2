from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = r"D:\Lama_project\TheBlokeLlama-2-13B-chat-AWQ"

model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
while True:
    prompt = input("enter your prompt:\n")
    prompt_template = f"[INST]{prompt}[/INST]"
    print("\n\n*** Generate:")

    tokens = tokenizer(
        prompt_template,
        return_tensors='pt'
    ).input_ids.cuda()


    generation_output = model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=1000
    )  
    print("Output: ", tokenizer.decode(generation_output[0]))