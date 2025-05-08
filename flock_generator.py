import torch
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from pattern_manager import PatternManager, SecurityConfig
import numpy as np
import time

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
    language_specialization: Optional[str] = None
    mandala_flow_alpha: float = 0.1 # Controls the speed of mandala transitions (0.0 to 1.0)
    mandala_shape_bias: float = 0.0 # Range -1.0 (bias to circles) to +1.0 (bias to triangles)

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
                      language: Optional[str] = None) -> Tuple[str, torch.Tensor]:
        """Generate text with pattern influence and language awareness, returning text and final states tensor."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get base generation
        with torch.no_grad():
            # Request scores to potentially get logits/hidden_states directly if needed, 
            # but model.generate output structure can vary. Here, 'outputs' are token IDs.
            generation_output = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True, # Ensure hidden states are output by generate if possible
                return_dict_in_generate=True, # Makes output more structured
                repetition_penalty=1.2 # Added to discourage repetition
            )
            # `outputs` from `generate` are typically token IDs. 
            # We need to get embeddings from these token IDs for further processing.
            output_token_ids = generation_output.sequences

        # Get hidden states from the model based on the generated token IDs
        with torch.no_grad():
            # Get input embeddings for the generated sequence
            # The shape of output_token_ids is [batch_size, sequence_length]
            # The model's embedding layer expects this shape.
            input_embeddings = self.model.get_input_embeddings()(output_token_ids)
            
            # Pass these embeddings through the model to get all hidden states
            model_output_from_tokens = self.model(
                inputs_embeds=input_embeddings,
                output_hidden_states=True
            )
            # The last hidden state corresponds to the features for each token in the generated sequence
            hidden_states = model_output_from_tokens.hidden_states[-1]
            
        # Apply pattern influence with language awareness
        # Ensure hidden_states has the expected 3D shape [batch, seq, hidden_dim]
        # If it came from generate() and is for a single sequence, it might be 2D [seq, hidden_dim]
        # or 3D [1, seq, hidden_dim]. PatternManager likely expects [batch, seq, hidden_dim].
        # Given output_token_ids is [batch_size, sequence_length], hidden_states should be [batch_size, sequence_length, hidden_dim]
        
        pattern_influenced = await self.pattern_manager.process_output(
            hidden_states, # This should be [batch, seq, dim]
            block_idx=0, # Placeholder, consider passing actual indices if relevant
            sequence_idx=0, # Placeholder
            maintain_consistency=True,
            language=language # Pass language here as well
        )
        
        # Apply language-specific processing if language is specified
        # Note: pattern_manager.process_output might already handle this if `language` is passed.
        # The original code had a separate call to _process_with_node. Let's assume
        # process_output is the main interface now for this step.
        # If _process_with_node is still essential and distinct, it would need to be called here.
        
        # Blend original and pattern-influenced outputs
        # Ensure hidden_states and pattern_influenced are broadcastable or match in shape.
        # If pattern_influenced changed shape (e.g. if PatternManager returns a different seq length or flattened), adjust here.
        final_states = (1 - pattern_influence) * hidden_states + pattern_influence * pattern_influenced
        
        # Convert back to tokens using output embeddings
        with torch.no_grad():
            # The logits should be calculated from final_states
            # final_states shape: [batch_size, sequence_length, hidden_dimension]
            logits = self.model.get_output_embeddings()(final_states) #This is LMHead for CausalLM
            final_tokens = logits.argmax(dim=-1) # Get token IDs [batch_size, sequence_length]
        
        # Decode only the first sequence in the batch for the string output, if batch_size > 1
        # The `final_states` tensor will contain the states for the whole batch.
        decoded_text = self.tokenizer.decode(final_tokens[0], skip_special_tokens=True)
        
        # Return the decoded text for the first item and the entire final_states tensor (for the batch)
        return decoded_text, final_states # final_states is [batch, seq_len, hidden_dim]

# Define the callback type
IterationCallback = Callable[[int, List[str], List[torch.Tensor]], None]

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
        
        self.memory_sync_interval = 5
        self.generation_count = 0
        self.last_sync_time = time.time()
        
    async def _apply_flock_behavior(self, 
                                  outputs: List[str], # Current texts
                                  patterns: List[torch.Tensor], # Tensors derived from current texts
                                  languages: List[str],
                                  iteration_num: int, # Current flocking iteration number
                                  iteration_callback: Optional[IterationCallback] = None
                                  ) -> List[str]: # Returns modified texts
        """Apply flocking behavior, with a callback for observing intermediate states."""
        # This method processes a list of texts, applies flocking to their tensor representations,
        # and then converts them back to text.
        
        # Convert current text outputs to hidden states (as before)
        hidden_states_for_flocking = []
        max_seq_len = 0
        for output_text in outputs: # outputs are current texts
            tokens = self.models[0].tokenizer(output_text, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.models[0].device)
            max_seq_len = max(max_seq_len, tokens.input_ids.size(1))
        
        for output_text in outputs:
            tokens = self.models[0].tokenizer(output_text, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.models[0].device)
            current_input_ids = tokens.input_ids
            with torch.no_grad():
                input_embeddings = self.models[0].model.get_input_embeddings()(current_input_ids)
                if input_embeddings.size(1) < max_seq_len:
                    padding_size = max_seq_len - input_embeddings.size(1)
                    padding = torch.zeros(input_embeddings.size(0), padding_size, input_embeddings.size(2)).to(input_embeddings.device)
                    input_embeddings = torch.cat([input_embeddings, padding], dim=1)
                elif input_embeddings.size(1) > max_seq_len:
                    input_embeddings = input_embeddings[:, :max_seq_len, :]
                
                model_output = self.models[0].model(inputs_embeds=input_embeddings, output_hidden_states=True)
                hidden_states_for_flocking.append(model_output.hidden_states[-1])

        # Variable to hold the state of hidden_states after flocking logic for this pass
        evolved_hidden_states_for_this_pass = list(hidden_states_for_flocking) # Start with a copy

        # Calculate flocking forces (applied to evolved_hidden_states_for_this_pass)
        for i in range(len(evolved_hidden_states_for_this_pass)):
            with torch.no_grad():
                lang_weight_i = self.config.language_weights.get(languages[i], 0.5)
                # The 'patterns' argument here could be `hidden_states_for_flocking` or pre-derived patterns
                # For simplicity, let's assume current_pattern is derived from evolved_hidden_states_for_this_pass[i]
                # If `patterns` arg was meant to be the *original* patterns for comparison, that needs clarification.
                # For now, using the current state to derive current_pattern for flocking dynamics.
                current_pattern_vec = self.pattern_manager.generate_pattern(evolved_hidden_states_for_this_pass[i])
                
                # Shared memory (original logic)
                if len(self.pattern_manager.shared_memory['patterns']) > 0:
                    similarities = self.pattern_manager._calculate_pattern_similarities(current_pattern_vec)
                    if similarities.numel() > 0:
                        max_sim_idx = similarities.argmax()
                        if similarities[max_sim_idx] > self.pattern_manager.borrow_threshold:
                            borrowed_pattern = self.pattern_manager.shared_memory['patterns'][max_sim_idx]
                            self.pattern_manager.shared_memory['usage_count'][max_sim_idx] += 1
                            blend_factor = similarities[max_sim_idx]
                            current_pattern_vec = (1 - blend_factor) * current_pattern_vec + blend_factor * borrowed_pattern
                
                # Cohesion, Separation, Alignment forces (applied to evolved_hidden_states_for_this_pass[i])
                center = torch.mean(torch.stack(evolved_hidden_states_for_this_pass), dim=0)
                cohesion_force = (center - evolved_hidden_states_for_this_pass[i]) * self.config.flock_cohesion * lang_weight_i
                
                separation_force = torch.zeros_like(evolved_hidden_states_for_this_pass[i])
                for j in range(len(evolved_hidden_states_for_this_pass)):
                    if i != j:
                        lang_weight_j = self.config.language_weights.get(languages[j], 0.5)
                        diff = evolved_hidden_states_for_this_pass[i] - evolved_hidden_states_for_this_pass[j]
                        separation_force += diff / (torch.norm(diff) + 1e-6) * (1 - abs(lang_weight_i - lang_weight_j))
                separation_force *= self.config.flock_separation

                alignment_force = torch.zeros_like(evolved_hidden_states_for_this_pass[i])
                total_weight = 0
                for j in range(len(evolved_hidden_states_for_this_pass)):
                    if i != j:
                        lang_weight_j = self.config.language_weights.get(languages[j], 0.5)
                        similarity = 1 - abs(lang_weight_i - lang_weight_j)
                        alignment_force += evolved_hidden_states_for_this_pass[j] * similarity
                        total_weight += similarity
                if total_weight > 0:
                    alignment_force = (alignment_force / total_weight - evolved_hidden_states_for_this_pass[i]) * self.config.flock_alignment
                
                evolved_hidden_states_for_this_pass[i] = evolved_hidden_states_for_this_pass[i].clone() + cohesion_force + separation_force + alignment_force
                
                # Apply pattern influence (this is pattern_manager.process_output)
                evolved_hidden_states_for_this_pass[i] = await self.pattern_manager.process_output(
                    evolved_hidden_states_for_this_pass[i],
                    block_idx=i,
                    sequence_idx=iteration_num, # Pass current flocking iteration
                    maintain_consistency=True,
                    language=languages[i]
                )
        
        # Convert evolved hidden states back to text for this iteration
        iter_texts_output = []
        for hidden_state in evolved_hidden_states_for_this_pass:
            with torch.no_grad():
                logits = self.models[0].model.get_output_embeddings()(hidden_state)
                tokens = logits.argmax(dim=-1)
                iter_texts_output.append(self.models[0].tokenizer.decode(tokens[0], skip_special_tokens=True))
            
        # Invoke callback with the state of this iteration BEFORE returning texts
        if iteration_callback:
            # The tensors for the callback are the evolved_hidden_states_for_this_pass
            iteration_callback(iteration_num, iter_texts_output, evolved_hidden_states_for_this_pass)
            
        return iter_texts_output # Return the modified texts
        
    async def generate_batch(self, 
                           prompts: List[str],
                           languages: Optional[List[str]] = None,
                           num_iterations: int = 3,
                           iteration_callback: Optional[IterationCallback] = None # Added callback
                           ) -> List[Tuple[str, torch.Tensor]]:
        """Generate text and tensors with flocking, providing iteration callbacks."""
        if languages is None:
            languages = [self.config.language_specialization] * len(prompts)
            
        initial_generation_tasks = []
        for i, (prompt, lang) in enumerate(zip(prompts, languages)):
            model_idx = i % len(self.models)
            initial_generation_tasks.append(
                self.models[model_idx].generate(
                    prompt, self.config.max_length, self.config.temperature,
                    self.config.top_p, language=lang
                )
            )
        initial_outputs_tuples = await asyncio.gather(*initial_generation_tasks)
        current_texts = [text for text, _ in initial_outputs_tuples]
        # We don't directly use the initial tensors in the loop if _apply_flock_behavior re-embeds texts.

        for i in range(num_iterations):
            # logger.debug(f"Flocking iteration {i + 1}/{num_iterations}")
            # Prepare patterns for _apply_flock_behavior (re-embedding current_texts)
            # This part remains somewhat inefficient but is necessary for current _apply_flock_behavior signature
            patterns_for_flocking = []
            # Simplified: assume _apply_flock_behavior will handle getting its own patterns or we pass None
            # Or, pass the tensors from the *previous* iteration if available and meaningful.
            # For now, let's let _apply_flock_behavior re-calculate from texts it receives.
            # The `patterns` argument to _apply_flock_behavior was used for its _extract_pattern logic.
            # Let's pass the `hidden_states` derived from `current_texts` as the `patterns` arg for consistency.
            # This means _apply_flock_behavior receives the *current* state to work with.
            # This is a bit circular if `patterns` arg is meant to be a target or distinct element.
            # However, the original logic used `patterns.append(pattern)` where `pattern` was from `get_input_embeddings`.
            # Let's skip re-calculating `patterns_for_flocking` here and assume `_apply_flock_behavior` will derive them from `current_texts` if needed
            # or that the `patterns` argument it receives is correctly used. 
            # The `_apply_flock_behavior` now re-embeds `outputs` (which are `current_texts`).
            # So, the `patterns` argument to `_apply_flock_behavior` might be redundant or used differently now.
            # Let's pass an empty list for `patterns` to `_apply_flock_behavior` for now, as it re-embeds `outputs` anyway.

            current_texts = await self._apply_flock_behavior(
                current_texts, 
                [], # Pass empty for `patterns` as it re-embeds `outputs`
                languages, 
                i + 1, # Iteration number (1-indexed)
                iteration_callback
            )
            
            self.generation_count += 1
            if self.generation_count % self.memory_sync_interval == 0:
                self._sync_memory()
        
        # Final tensor calculation (as before)
        final_results_tuples: List[Tuple[str, torch.Tensor]] = []
        for idx, final_text in enumerate(current_texts):
            target_model = self.models[idx % len(self.models)]
            lang = languages[idx]
            inputs = target_model.tokenizer(final_text, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(target_model.device)
            output_token_ids = inputs.input_ids
            with torch.no_grad():
                input_embeddings = target_model.model.get_input_embeddings()(output_token_ids)
                model_output_from_tokens = target_model.model(inputs_embeds=input_embeddings, output_hidden_states=True)
                hidden_states = model_output_from_tokens.hidden_states[-1]
            
            # Using a fixed pattern_influence for the final step. This could be a config.
            pattern_influence_final = 0.5 
            pattern_influenced = await target_model.pattern_manager.process_output(
                hidden_states, block_idx=idx, sequence_idx=num_iterations, 
                maintain_consistency=True, language=lang
            )
            final_states = (1 - pattern_influence_final) * hidden_states + pattern_influence_final * pattern_influenced
            final_results_tuples.append((final_text, final_states))
            
        return final_results_tuples

    def _sync_memory(self):
        """Synchronize memory across all models."""
        # Get memory stats before sync
        pre_sync_stats = self.pattern_manager.get_memory_stats()
        
        # Update memory decay based on time elapsed
        time_elapsed = time.time() - self.last_sync_time
        decay_factor = self.pattern_manager.memory_decay ** time_elapsed
        
        # Apply decay to all patterns
        self.pattern_manager.shared_memory['affinity_scores'] = [
            score * decay_factor for score in self.pattern_manager.shared_memory['affinity_scores']
        ]
        
        # Remove patterns with very low affinity
        low_affinity_indices = [
            i for i, score in enumerate(self.pattern_manager.shared_memory['affinity_scores'])
            if score < 0.1
        ]
        for key in self.pattern_manager.shared_memory:
            self.pattern_manager.shared_memory[key] = [
                val for i, val in enumerate(self.pattern_manager.shared_memory[key])
                if i not in low_affinity_indices
            ]
        
        # Update last sync time
        self.last_sync_time = time.time()
        
        # Get memory stats after sync
        post_sync_stats = self.pattern_manager.get_memory_stats()
        
        # Log memory sync results
        logger.info(f"Memory sync completed:")
        logger.info(f"Patterns before: {pre_sync_stats['total_patterns']}")
        logger.info(f"Patterns after: {post_sync_stats['total_patterns']}")
        logger.info(f"Memory utilization: {post_sync_stats['memory_utilization']:.2%}")

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