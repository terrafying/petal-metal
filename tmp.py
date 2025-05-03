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

# Configure DHT with explicit transport settings
dht = DHT(
    initial_peers=PEERS,
    start=True,
    client_mode=True,
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip6/::/tcp/0"],  # Listen on all interfaces
    announce_maddrs=[],  # Don't announce any addresses
    use_relay=False,  # Disable relay to avoid complexity
    use_auto_relay=False,  # Disable auto relay
    startup_timeout=30,  # Shorter timeout for faster feedback
)

infos = get_remote_module_infos(dht)
for info in infos:
    print(info.block_id, "→", info.peer_id)

config = ClientConfig(
    initial_peers=PEERS,
    show_route="inference",
    use_relay=False,
    use_auto_relay=False,
    daemon_startup_timeout=30,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL,
    config=config,
    torch_dtype="float32",
    device_map={"": "mps"},
)

inputs = tokenizer("Hello!", return_tensors="pt").to("mps")
out = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(out[0], skip_special_tokens=True)) 