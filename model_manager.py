import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Union
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig

# Configure logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model downloads, conversion, and updates."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/petals"))
        self.models_dir = self.cache_dir / "models"
        self.config_file = self.cache_dir / "model_config.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._load_config()
        
    def _load_config(self):
        """Load model configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "models": {},
                "active_model": None
            }
            self._save_config()
            
    def _save_config(self):
        """Save model configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def list_models(self) -> List[Dict]:
        """List all available models."""
        return [
            {
                "name": name,
                "path": info["path"],
                "version": info.get("version", "unknown"),
                "size": info.get("size", 0),
                "is_active": name == self.config["active_model"]
            }
            for name, info in self.config["models"].items()
        ]
        
    def download_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        force: bool = False
    ) -> str:
        """Download and convert a model."""
        model_dir = self.models_dir / model_name.replace("/", "_")
        
        if model_dir.exists() and not force:
            logger.info(f"Model {model_name} already exists at {model_dir}")
            return str(model_dir)
            
        # Create temporary directory for download
        temp_dir = self.cache_dir / f"temp_{model_name.replace('/', '_')}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download model files
            logger.info(f"Downloading model {model_name}...")
            snapshot_download(
                repo_id=model_name,
                revision=revision,
                token=token,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            
            # Get model config
            config = AutoConfig.from_pretrained(temp_dir)
            model_size = sum(
                f.stat().st_size for f in temp_dir.rglob("*")
                if f.is_file()
            )
            
            # Convert model if needed
            if not self._is_converted(temp_dir):
                logger.info("Converting model...")
                self._convert_model(temp_dir)
                
            # Move to final location
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.move(temp_dir, model_dir)
            
            # Update config
            self.config["models"][model_name] = {
                "path": str(model_dir),
                "version": revision or "latest",
                "size": model_size,
                "config": config.to_dict()
            }
            self._save_config()
            
            logger.info(f"Successfully downloaded and converted model {model_name}")
            return str(model_dir)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
            
    def _is_converted(self, model_dir: Path) -> bool:
        """Check if model is already converted."""
        # Check for converted model files
        return (model_dir / "config.json").exists() and (
            (model_dir / "pytorch_model.bin").exists() or
            (model_dir / "model.safetensors").exists()
        )
        
    def _convert_model(self, model_dir: Path):
        """Convert model to Petals format."""
        # Import here to avoid circular imports
        from petals.cli.convert_model import convert_model
        
        # Convert model
        convert_model(
            model_dir,
            output_dir=model_dir,
            device="cpu"
        )
        
    def set_active_model(self, model_name: str):
        """Set the active model."""
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found")
            
        self.config["active_model"] = model_name
        self._save_config()
        logger.info(f"Set active model to {model_name}")
        
    def get_active_model(self) -> Optional[Dict]:
        """Get the active model info."""
        if not self.config["active_model"]:
            return None
            
        model_name = self.config["active_model"]
        return {
            "name": model_name,
            **self.config["models"][model_name]
        }
        
    def remove_model(self, model_name: str):
        """Remove a model."""
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found")
            
        model_dir = Path(self.config["models"][model_name]["path"])
        if model_dir.exists():
            shutil.rmtree(model_dir)
            
        del self.config["models"][model_name]
        if self.config["active_model"] == model_name:
            self.config["active_model"] = None
        self._save_config()
        
        logger.info(f"Removed model {model_name}")
        
    def update_model(self, model_name: str, token: Optional[str] = None):
        """Update a model to the latest version."""
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found")
            
        return self.download_model(
            model_name,
            token=token,
            force=True
        ) 