

python3 -m petals.cli.run_server \
                 --initial_peers ip4/192.168.1.223/tcp/31330/p2p/12D3KooWK6ahZsAKjEk6QmiZ787c3FC17jLj6MCQgtsR9jwnvQK1 \
                 --port 31331 \
                 --converted_model_name_or_path <HF_MODEL_ID> \
                 --device mps \
                 --torch_dtype float16
