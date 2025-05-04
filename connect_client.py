import logging
import time
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from unified_discovery import UnifiedDiscovery
import traceback

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG  # Set to DEBUG for more detailed logs
)
logger = logging.getLogger(__name__)

class PetalsClient:
    """Helper class to connect to Petals servers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.discovery = UnifiedDiscovery()
        self.model = None
        self.tokenizer = None
        
    def connect(self, wait_for_peers: bool = True, timeout: int = 30) -> bool:
        """
        Connect to Petals servers.
        
        Args:
            wait_for_peers: Whether to wait for peers to be discovered
            timeout: Maximum time to wait for peers in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Start discovery
            logger.debug("Starting peer discovery...")
            self.discovery.start_discovery()
            
            # Wait for peers if requested
            if wait_for_peers:
                logger.debug(f"Waiting for peers (timeout: {timeout}s)...")
                start_time = time.time()
                while time.time() - start_time < timeout:
                    peers = self.discovery.get_peers()
                    if peers:
                        logger.info(f"Found {len(peers)} peers: {peers}")
                        break
                    time.sleep(1)
            else:
                peers = self.discovery.get_peers()
            
            if peers:
                logger.info(f"Found peers: {peers}")
                
                # Initialize model and tokenizer
                logger.debug(f"Initializing tokenizer for model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                logger.debug("Initializing distributed model...")
                try:
                    self.model = AutoDistributedModelForCausalLM.from_pretrained(
                        self.model_name,
                        initial_peers=peers,
                        torch_dtype=torch.float32,
                        device_map={"": "mps"},
                    )
                    logger.info("Successfully initialized distributed model")
                    return True
                except Exception as model_error:
                    logger.error(f"Failed to initialize model: {str(model_error)}")
                    logger.debug(f"Model initialization error details:\n{traceback.format_exc()}")
                    return False
            else:
                logger.warning(f"No peers found after {timeout}s of waiting")
                return False
                
        except Exception as e:
            logger.error(f"Error during connection: {str(e)}")
            logger.debug(f"Connection error details:\n{traceback.format_exc()}")
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
        """Close the connection and cleanup resources."""
        if self.discovery:
            self.discovery.stop()
        self.model = None
        self.tokenizer = None

def main():
    """Example usage of the PetalsClient."""
    client = PetalsClient("bigscience/bloom-7b1-petals")
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