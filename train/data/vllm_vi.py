## Install vllm another python env 
## uv pip install vllm --torch-backend=auto
from transformers import pipeline
from vllm import LLM, SamplingParams
import datasets
import pandas as pd

dataset = datasets.load_dataset('csv', data_files="dialog-acts+slots.csv")
def add_prompt(example):
  example["prompt"] =  [
    {"role": "user", "content": "Dịch ra tiếng Việt câu hội thoại tiếng Anh sau: " + example['utterance']}
  ]
  return example
temp_data = dataset.map(add_prompt)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64
)

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

results = llm.chat(
    messages=temp_data["train"]["prompt"][:],
    sampling_params=sampling_params,
    use_tqdm=True
)

generated_texts = [output.outputs[0].text for output in results]

df = dataset["train"].to_pandas()
df2 = df[["label", "dataset"]]
df2["utterance"] = generated_texts
df = pd.concat([df, df2], ignore_index=True)
df.to_csv("ev-dialog-acts+slots.csv", index=False)
