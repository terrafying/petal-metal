python3 -m petals.cli.run_server \
                 --new_swarm \
                 --port 31330 \
                 --converted_model_name_or_path "bigscience/bloom-7b1-petals" \
                 --device mps \
                 --torch_dtype float16
# This will run the server on the local machine, and you can connect to it using the connect_client.py file.
