python3 -m petals.cli.run_server \
                 --new_swarm \
                 --port 31330 \
                 --converted_model_name_or_path "bigscience/bloom-7b1-petals" \
                 --device mps \
                 --torch_dtype float16
