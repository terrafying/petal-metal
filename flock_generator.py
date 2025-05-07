import torch
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager, SecurityConfig
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FlockConfig:
    num_models: int = 8
    batch_size: int = 32
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    pattern_depth: int = 3
    flock_cohesion: float = 0.5  # How strongly models influence each other
    flock_separation: float = 0.3  # How much models try to stay distinct
    flock_alignment: float = 0.4  # How much models try to align their outputs
    language_weights: Dict[str, float] = field(default_factory=lambda: {
        "en": 1.0,
        "es": 0.8,
        "fr": 0.8,
        "de": 0.8,
        "zh": 0.7,
        "ja": 0.7,
        "ko": 0.7,
        "ru": 0.8,
        "ar": 0.7,
        "hi": 0.7
    })
    language_specialization: str = "en"

class FlockModel:
    def __init__(self, 
                 model_name: str,
                 language: str = "en",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.language = language
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pattern_manager = PatternManager(
            model=self.model,
            pattern_depth=3,
            base_seed=42
        )
        
    async def generate(self, 
                      prompt: str,
                      max_length: int,
                      temperature: float,
                      top_p: float,
                      pattern_influence: float = 0.5,
                      language: Optional[str] = None) -> str:
        """Generate text with pattern influence and language awareness."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get base generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Get hidden states from the model
        with torch.no_grad():
            # Get input embeddings
            input_embeddings = self.model.get_input_embeddings()(outputs)
            
            # Pass through model to get hidden states
            model_output = self.model(
                inputs_embeds=input_embeddings,
                output_hidden_states=True
            )
            hidden_states = model_output.hidden_states[-1]  # Get the last layer's hidden states
            
        # Apply pattern influence with language awareness
        pattern_influenced = await self.pattern_manager.process_output(
            hidden_states,
            block_idx=0,
            sequence_idx=0,
            maintain_consistency=True
        )
        
        # Apply language-specific processing if language is specified
        if language:
            pattern_influenced = self.pattern_manager._process_with_node(
                node_id=0,  # Use first node for language processing
                tensor=pattern_influenced,
                block_idx=0,
                sequence_idx=0,
                language=language
            )
        
        # Blend original and pattern-influenced outputs
        final_states = (1 - pattern_influence) * hidden_states + pattern_influence * pattern_influenced
        
        # Convert back to tokens using output embeddings
        with torch.no_grad():
            logits = self.model.get_output_embeddings()(final_states)
            final_tokens = logits.argmax(dim=-1)
        
        return self.tokenizer.decode(final_tokens[0], skip_special_tokens=True)

class FlockGenerator:
    def __init__(self, 
                 model_names: List[str],
                 config: Optional[FlockConfig] = None):
        self.config = config or FlockConfig()
        self.models = [
            FlockModel(
                name,
                language=lang,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ) for name, lang in zip(
                model_names[:self.config.num_models],
                [self.config.language_specialization] * self.config.num_models
            )
        ]
        self.pattern_manager = PatternManager(
            model=self.models[0].model,
            pattern_depth=self.config.pattern_depth,
            base_seed=42
        )
        
    async def _apply_flock_behavior(self, 
                                  outputs: List[str],
                                  patterns: List[torch.Tensor],
                                  languages: List[str]) -> List[str]:
        """Apply flocking behavior to model outputs with language awareness."""
        # Convert outputs to hidden states through the model
        hidden_states = []
        max_seq_len = 0
        
        # First pass: get max sequence length
        for output in outputs:
            tokens = self.models[0].tokenizer(output, return_tensors="pt")
            max_seq_len = max(max_seq_len, tokens.input_ids.size(1))
        
        # Second pass: process all outputs to same length
        for output in outputs:
            # Tokenize input
            tokens = self.models[0].tokenizer(output, return_tensors="pt")
            
            with torch.no_grad():
                # Get input embeddings
                input_embeddings = self.models[0].model.get_input_embeddings()(tokens.input_ids)
                
                # Pad or truncate to max_seq_len
                if input_embeddings.size(1) < max_seq_len:
                    # Pad with zeros
                    padding = torch.zeros(1, max_seq_len - input_embeddings.size(1), input_embeddings.size(2))
                    input_embeddings = torch.cat([input_embeddings, padding], dim=1)
                elif input_embeddings.size(1) > max_seq_len:
                    # Truncate
                    input_embeddings = input_embeddings[:, :max_seq_len, :]
                
                # Pass through model layers to get hidden states
                model_output = self.models[0].model(
                    inputs_embeds=input_embeddings,
                    output_hidden_states=True
                )
                hidden_states.append(model_output.hidden_states[-1])
            
        # Calculate flocking forces with language weights
        for i in range(len(hidden_states)):
            with torch.no_grad():  # Ensure consistent gradient handling
                # Get language weights
                lang_weight_i = self.config.language_weights.get(languages[i], 0.5)
                
                # Cohesion: Move towards center of mass with language weighting
                center = torch.mean(torch.stack(hidden_states), dim=0)
                cohesion_force = (center - hidden_states[i]) * self.config.flock_cohesion * lang_weight_i
                
                # Separation: Move away from neighbors with language awareness
                separation_force = torch.zeros_like(hidden_states[i])
                for j in range(len(hidden_states)):
                    if i != j:
                        lang_weight_j = self.config.language_weights.get(languages[j], 0.5)
                        diff = hidden_states[i] - hidden_states[j]
                        separation_force += diff / (torch.norm(diff) + 1e-6) * (1 - abs(lang_weight_i - lang_weight_j))
                separation_force *= self.config.flock_separation
                
                # Alignment: Align with neighbors of similar languages
                alignment_force = torch.zeros_like(hidden_states[i])
                total_weight = 0
                for j in range(len(hidden_states)):
                    if i != j:
                        lang_weight_j = self.config.language_weights.get(languages[j], 0.5)
                        similarity = 1 - abs(lang_weight_i - lang_weight_j)
                        alignment_force += hidden_states[j] * similarity
                        total_weight += similarity
                if total_weight > 0:
                    alignment_force = (alignment_force / total_weight - hidden_states[i]) * self.config.flock_alignment
                
                # Apply forces
                hidden_states[i] = hidden_states[i].clone() + cohesion_force + separation_force + alignment_force
                
                # Apply pattern influence with language awareness
                hidden_states[i] = await self.pattern_manager.process_output(
                    hidden_states[i],
                    block_idx=i,
                    sequence_idx=0,
                    maintain_consistency=True,
                    language=languages[i]
                )
            
        # Convert back to text through output embeddings
        results = []
        for hidden_state in hidden_states:
            with torch.no_grad():
                # Get logits from output embeddings
                # Shape: (batch_size, sequence_length, vocab_size)
                logits = self.models[0].model.get_output_embeddings()(hidden_state)
                
                # Get most likely tokens
                tokens = logits.argmax(dim=-1)
                
                # Decode tokens to text
                results.append(self.models[0].tokenizer.decode(tokens[0], skip_special_tokens=True))
            
        return results
        
    async def generate_batch(self, 
                           prompts: List[str],
                           languages: Optional[List[str]] = None,
                           num_iterations: int = 3) -> List[str]:
        """Generate text in parallel with flocking behavior and language support."""
        if languages is None:
            languages = [self.config.language_specialization] * len(prompts)
            
        # Initial generation
        tasks = []
        for i, (prompt, lang) in enumerate(zip(prompts, languages)):
            model_idx = i % len(self.models)
            tasks.append(
                self.models[model_idx].generate(
                    prompt,
                    self.config.max_length,
                    self.config.temperature,
                    self.config.top_p,
                    language=lang
                )
            )
        
        outputs = await asyncio.gather(*tasks)
        
        # Apply flocking behavior iteratively
        for _ in range(num_iterations):
            # Get patterns for each output
            patterns = []
            for output in outputs:
                tokens = self.models[0].tokenizer(output, return_tensors="pt")
                with torch.no_grad():
                    pattern = self.models[0].model.get_input_embeddings()(tokens.input_ids)
                patterns.append(pattern)
            
            # Apply flocking behavior with language awareness
            outputs = await self._apply_flock_behavior(outputs, patterns, languages)
            
        return outputs

async def main():
    # Example usage
    model_names = [
        "gpt2",
        "distilgpt2",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B"
    ]
    
    config = FlockConfig(
        num_models=4,
        batch_size=8,
        max_length=256,
        temperature=0.8,
        flock_cohesion=0.6,
        flock_separation=0.4,
        flock_alignment=0.5
    )
    
    generator = FlockGenerator(model_names, config)
    
    prompts = [
        "Once upon a time",
        "In a world where",
        "The future of AI",
        "When machines learn"
    ] * 2  # 8 prompts total
    
    results = await generator.generate_batch(prompts, num_iterations=3)
    
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 