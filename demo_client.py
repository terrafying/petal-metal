from connect_client import PetalsClient

# Initialize and connect to available peers
client = PetalsClient(model_name="bigscience/bloom-7b1-petals")
try:
    if client.connect():
        # Generate text
        response = client.generate("Hello, how are you?")
        print("Generated response:")
        print(response)
finally:
    client.close()
    print("Client connection closed.") 