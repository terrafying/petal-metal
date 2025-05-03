from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from petals.client.config import ClientConfig

MODEL = "bigscience/bloom-7b1-petals"
PEERS = [
    "/ip4/192.168.1.198/tcp/31337/p2p/12D3KooWMQomr2FZcais1XhnwFCbY2uFmQh2qbGJaoXykfmSUfFH",  # Mac A
    "/ip4/192.168.1.223/tcp/31337/p2p/12D3KooWNmsW2o8P9UUbxYWzav9jQuqgpi3ExZ77gm8vjxRmkSMT",   # Mac B
]

from hivemind import DHT
from petals.utils.dht import get_remote_module_infos

dht = DHT(initial_peers=PEERS, start=True)
infos = get_remote_module_infos(dht)
for info in infos:
    # info.block_id is e.g. "bigscience/bloom-7b1-petals.3"
    # info.peer_id is the PeerID string
    print(info.block_id, "→", info.peer_id)
config = ClientConfig(
    initial_peers=PEERS,
    show_route="inference",   # ← prints the route
    # skip_reachability_check=True,  # if you’re still using that hack
    # no_relay=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL,
    config=config,
    torch_dtype="float32",
    device_map={"": "mps"},
)

inputs = tokenizer("Hello!", return_tensors="pt").to("mps")
out    = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(out[0], skip_special_tokens=True))


from hivemind import DHT
from petals.utils.dht import get_remote_module_infos

dht = DHT(initial_peers=PEERS, start=True)
infos = get_remote_module_infos(dht, prefix="bigscience/bloom-7b1-petals")
for info in infos:
    # info.block_id is e.g. "bigscience/bloom-7b1-petals.3"
    # info.peer_id is the PeerID string
    print(info.block_id, "→", info.peer_id)
