
import os, torch
from transformers import AutoTokenizer
from petals import DistributedBloomForCausalLM

os.environ["PETALS_SERVERS"] = ",".join([
  "minipro24.local:31337",
  "/ip4/192.168.1.223/tcp/31337/p2p/12D3KooWNmsW2o8P9UUbxYWzav9jQuqgpi3ExZ77gm8vjxRmkSMT",
  # …etc
])

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")

model = DistributedBloomForCausalLM.from_pretrained(
  "bigscience/bloomz-7b1",
  torch_dtype=torch.float16,
  device_map={"": "mps"},       # put your final generation steps onto the GPU
)

inputs = tokenizer("Hello Metal!", return_tensors="pt").to("mps")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))

