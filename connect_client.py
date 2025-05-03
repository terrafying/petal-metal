from simple_discovery import SimplePeerDiscovery
from process_lock import PetalsProcessLock
from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PetalsClient:
    """Helper class to connect to Petals servers using service discovery."""
    
    def __init__(self, model_name: str = "bigscience/bloom-7b1-petals"):
        self.model_name = model_name
        self.discovery = SimplePeerDiscovery()
        self.process_lock = PetalsProcessLock()
        self.model = None
        self.tokenizer = None
        
    def connect(self, wait_for_peers: bool = True, timeout: int = 30) -> bool:
        """
        Connect to available Petals servers.
        
        Args:
            wait_for_peers: If True, wait for peers to be discovered
            timeout: Maximum time to wait for peers in seconds
            
        Returns:
            bool: True if connection successful
        """
        # Try to acquire the process lock with a shorter timeout and less strict resource checks
        if not self.process_lock.acquire(timeout=5, check_resource_limits=False):
            logger.warning("Another Petals process is running, proceeding with caution")
        
        logger.info("Starting service discovery...")
        self.discovery.start_discovery()
        
        if wait_for_peers:
            start_time = time.time()
            while time.time() - start_time < timeout:
                peers = self.discovery.get_peers(self.model_name)
                if peers:
                    break
                logger.info("Waiting for peers...")
                time.sleep(2)
            
            if not peers:
                logger.error(f"No peers found after {timeout} seconds")
                return False
        else:
            peers = self.discovery.get_peers(self.model_name)
        
        if peers:
            logger.info(f"Found peers: {peers}")
            
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoDistributedModelForCausalLM.from_pretrained(
                self.model_name,
                initial_peers=peers,
                dht_prefix=self.model_name,
                torch_dtype=torch.float16,
                device_map={"": "mps"},
            )
            
            return True
        else:
            logger.warning("No peers found")
            return False
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """
        Generate text using the connected model.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Not connected to any peers. Call connect() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("mps")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def close(self):
        """Clean up resources."""
        if self.discovery:
            self.discovery.stop()
        if self.process_lock:
            self.process_lock.release()
        # Model and tokenizer cleanup if needed
        self.model = None
        self.tokenizer = None

def main():
    """Example usage of the PetalsClient."""
    client = PetalsClient()
    try:
        if client.connect():
            # Example generation
            response = client.generate("Hello, how are you?")
            print(f"Generated response: {response}")
        else:
            print("Failed to connect to any peers")
    finally:
        client.close()

if __name__ == "__main__":
    main() 