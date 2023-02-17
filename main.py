import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

model = GPT2LMHeadModel(GPT2Config(n_positions=2048))
model.transformer.wte = torch.nn.Embedding(50264, 768)
model.lm_head = torch.nn.Linear(768, 50264, bias=False)
model.load_state_dict(torch.load('trained_on_dune.pt', map_location=torch.device('cpu')))

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

model.eval()

def generate(prompt):
        prompt = tokenizer.encode(prompt, return_tensors='pt')
        out = model.generate(
            input_ids=prompt,
            max_length=150,
            num_beams=5,
            do_sample=True,
            temperature=1,
            top_k=50,
            top_p=0.6,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            ).numpy()
        return tokenizer.decode(out[0])
demo = gr.Interface(fn=generate,
    inputs=gr.inputs.Textbox(label="Input Text"),
    outputs=gr.outputs.Textbox(label="Generated Text"))
demo.launch(share=True)
