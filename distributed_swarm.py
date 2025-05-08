import ray
import torch
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager
from flock_generator import FlockConfig, FlockGenerator
from storyteller import Storyteller
from visualization_manager import VisualizationManager
import logging
import time
import numpy as np
import soundfile as sf
from scipy import signal
import librosa

logger = logging.getLogger(__name__)

@ray.remote
class ModelWorker:
    """Ray actor for model inference."""
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate(self, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        """Generate text using the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

@ray.remote
class PatternWorker:
    """Ray actor for pattern processing."""
    def __init__(self, pattern_depth: int = 3):
        self.pattern_depth = pattern_depth
        
    def process_pattern(self, text: str, model_worker: ModelWorker) -> Dict:
        """Process text into a pattern."""
        # Get embeddings
        tokens = model_worker.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            embeddings = model_worker.model.get_input_embeddings()(tokens.input_ids)
            pattern = embeddings.mean(dim=1)  # Average over sequence length
            
        return {
            "pattern": pattern.cpu().numpy(),
            "text": text,
            "timestamp": time.time()
        }

@ray.remote
class VisualizationWorker:
    """Ray actor for visualization generation."""
    def __init__(self):
        self.viz_manager = VisualizationManager()
        
    def create_visualization(self, pattern: Dict) -> Dict:
        """Create visualizations from pattern."""
        mandala = self.viz_manager.create_mandala(pattern["pattern"])
        concretion = self.viz_manager.create_concretion(pattern["pattern"])
        
        return {
            "mandala": mandala,
            "concretion": concretion,
            "pattern": pattern
        }

@ray.remote
class StorytellerWorker:
    """Ray actor for narrative generation."""
    def __init__(self):
        self.storyteller = Storyteller(None)  # Pattern manager not needed for basic operations
        
    def add_thread(self, pattern: Dict, visualization: Dict) -> None:
        """Add a new narrative thread."""
        self.storyteller.add_thread(
            pattern=torch.tensor(pattern["pattern"]),
            mandala=visualization["mandala"],
            concretion=visualization["concretion"],
            context={"timestamp": pattern["timestamp"]}
        )
        
    def generate_narrative(self) -> Dict:
        """Generate narrative from threads."""
        return self.storyteller.generate_narrative()

@ray.remote
class AudioWorker:
    """Ray actor for audio generation."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_frequencies = {
            "harmony": 440.0,  # A4
            "transformation": 523.25,  # C5
            "resonance": 659.25,  # E5
            "emergence": 783.99,  # G5
            "synthesis": 987.77,  # B5
        }
        
    def generate_soundscape(self, pattern: Dict, emotional_valence: float, theme: str) -> Dict:
        """Generate a soundscape from pattern and emotional data."""
        # Extract pattern features
        pattern_array = pattern["pattern"]
        duration = 10.0  # seconds
        
        # Generate base frequency from theme
        base_freq = self.base_frequencies.get(theme, 440.0)
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate harmonic series based on pattern
        harmonics = []
        for i in range(5):
            freq = base_freq * (i + 1)
            amplitude = np.abs(pattern_array[i]) if i < len(pattern_array) else 0.5
            phase = np.angle(pattern_array[i]) if i < len(pattern_array) else 0
            
            # Modulate frequency with emotional valence
            freq_mod = freq * (1 + emotional_valence * 0.2)
            
            # Generate harmonic
            harmonic = amplitude * np.sin(2 * np.pi * freq_mod * t + phase)
            harmonics.append(harmonic)
        
        # Mix harmonics
        soundscape = np.sum(harmonics, axis=0)
        
        # Normalize
        soundscape = soundscape / np.max(np.abs(soundscape))
        
        # Apply emotional envelope
        envelope = np.exp(-t * (1 - emotional_valence))
        soundscape = soundscape * envelope
        
        # Add reverb
        reverb = signal.convolve(soundscape, np.exp(-t[:int(self.sample_rate/2)]), mode='same')
        soundscape = 0.7 * soundscape + 0.3 * reverb
        
        # Save to file
        filename = f"soundscape_{int(time.time())}.wav"
        sf.write(filename, soundscape, self.sample_rate)
        
        return {
            "filename": filename,
            "duration": duration,
            "base_frequency": base_freq,
            "emotional_valence": emotional_valence,
            "theme": theme
        }
        
    def generate_mantra_audio(self, mantra: str, pattern: Dict) -> Dict:
        """Generate audio for a specific mantra."""
        # Convert mantra to frequency sequence
        frequencies = []
        for char in mantra:
            # Map character to frequency using ASCII value
            freq = 220 * (1 + ord(char) / 128)
            frequencies.append(freq)
        
        # Generate time array
        duration = 5.0  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate sound
        sound = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            # Create time window for this frequency
            window_start = i * duration / len(frequencies)
            window_end = (i + 1) * duration / len(frequencies)
            window = (t >= window_start) & (t < window_end)
            
            # Generate tone with pattern influence
            amplitude = np.abs(pattern["pattern"][i % len(pattern["pattern"])])
            phase = np.angle(pattern["pattern"][i % len(pattern["pattern"])])
            tone = amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Apply window
            sound[window] = tone[window]
        
        # Normalize
        sound = sound / np.max(np.abs(sound))
        
        # Save to file
        filename = f"mantra_{int(time.time())}.wav"
        sf.write(filename, sound, self.sample_rate)
        
        return {
            "filename": filename,
            "duration": duration,
            "frequencies": frequencies,
            "mantra": mantra
        }

async def distributed_pattern_evolution():
    """Run pattern evolution swarm using Ray for distributed execution."""
    # Initialize Ray
    ray.init()
    
    # Initialize workers
    model_workers = [
        ModelWorker.remote("distilgpt2"),
        ModelWorker.remote("gpt2")
    ]
    pattern_worker = PatternWorker.remote()
    viz_worker = VisualizationWorker.remote()
    storyteller_worker = StorytellerWorker.remote()
    audio_worker = AudioWorker.remote()
    
    # Mantric tantagrams for pattern seeding
    mantric_tantagrams = ["Om", "Om Namah Shivaya"]
    
    # Pattern evolution loop
    for i in range(2):
        logger.info(f"Starting generation {i+1}/2")
        
        # Generate patterns in parallel
        pattern_futures = []
        for mantra in mantric_tantagrams:
            for model_worker in model_workers:
                # Generate text
                text_future = model_worker.generate.remote(
                    mantra,
                    max_length=64,
                    temperature=0.7,
                    top_p=0.9
                )
                # Process pattern
                pattern_future = pattern_worker.process_pattern.remote(
                    ray.get(text_future),
                    model_worker
                )
                pattern_futures.append(pattern_future)
        
        # Get patterns
        patterns = ray.get(pattern_futures)
        
        # Create visualizations in parallel
        viz_futures = [
            viz_worker.create_visualization.remote(pattern)
            for pattern in patterns
        ]
        visualizations = ray.get(viz_futures)
        
        # Generate audio in parallel
        audio_futures = []
        for pattern, viz in zip(patterns, visualizations):
            # Generate soundscape
            soundscape_future = audio_worker.generate_soundscape.remote(
                pattern,
                viz["pattern"]["emotional_valence"],
                viz["pattern"]["theme"]
            )
            # Generate mantra audio
            mantra_future = audio_worker.generate_mantra_audio.remote(
                pattern["text"],
                pattern
            )
            audio_futures.extend([soundscape_future, mantra_future])
        
        audio_outputs = ray.get(audio_futures)
        
        # Add to storyteller and generate narrative
        for pattern, viz in zip(patterns, visualizations):
            storyteller_worker.add_thread.remote(pattern, viz)
        
        narrative = ray.get(storyteller_worker.generate_narrative.remote())
        
        # Log narrative stats
        logger.info(f"Narrative stats for generation {i+1}:")
        logger.info(f"Themes: {narrative['themes']}")
        logger.info(f"Emotional range: {narrative['emotional_arc']}")
        
        # Save narrative and audio info to file
        with open(f'narrative_generation_{i+1}.txt', 'w') as f:
            f.write("=== Narrative ===\n\n")
            f.write(narrative["narrative"])
            f.write("\n\n=== Themes ===\n")
            f.write(", ".join(narrative["themes"]))
            f.write("\n\n=== Emotional Arc ===\n")
            f.write(", ".join(f"{v:.2f}" for v in narrative["emotional_arc"]))
            f.write("\n\n=== Visual Elements ===\n")
            for element in narrative["visual_elements"]:
                f.write(f"\nTheme: {element['theme']}\n")
                f.write(f"Emotional Valence: {element['emotional_valence']:.2f}\n")
                f.write(f"Mandala:\n{element['mandala']}\n")
                f.write(f"Concretion:\n{element['concretion']}\n")
            
            f.write("\n\n=== Audio Elements ===\n")
            for audio in audio_outputs:
                f.write(f"\nFile: {audio['filename']}\n")
                if 'mantra' in audio:
                    f.write(f"Mantra: {audio['mantra']}\n")
                    f.write(f"Frequencies: {audio['frequencies']}\n")
                else:
                    f.write(f"Theme: {audio['theme']}\n")
                    f.write(f"Base Frequency: {audio['base_frequency']}\n")
                    f.write(f"Emotional Valence: {audio['emotional_valence']:.2f}\n")
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Distributed pattern evolution completed")

if __name__ == "__main__":
    asyncio.run(distributed_pattern_evolution()) 