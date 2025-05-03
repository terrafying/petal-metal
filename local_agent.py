from typing import Dict, Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, Tool

class LocalAgent:
    """
    A local agent that uses a language model to interpret tasks and execute them using available tools.
    """
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        run_prompt_template: Optional[str] = None,
        additional_tools: Optional[Dict[str, Tool]] = None
    ):
        """
        Initialize the LocalAgent.
        
        Args:
            model: The language model to use for generating responses
            tokenizer: The tokenizer for the model
            run_prompt_template: Optional template for formatting prompts
            additional_tools: Optional dictionary of additional tools to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.run_prompt_template = run_prompt_template
        self._toolbox = additional_tools or {}

    def run(self, task: str, **kwargs: Any) -> Any:
        """
        Run a task using the language model and available tools.
        
        Args:
            task: The task description
            **kwargs: Additional arguments to pass to the task
        
        Returns:
            The result of executing the task
        """
        # Format the prompt using the template if provided
        if self.run_prompt_template:
            # Replace placeholders in the template
            tools_desc = "\n".join(f"- {name}: {tool.description}" for name, tool in self._toolbox.items())
            prompt = self.run_prompt_template.replace("<<all_tools>>", tools_desc)
            prompt = prompt + f"\nTask: {task}\n"
        else:
            prompt = task

        # Add any additional context from kwargs
        for key, value in kwargs.items():
            prompt += f"\n{key}: {value}"

        # Generate response using the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,  # Adjust as needed
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse the response and execute tools
        # This is a simplified implementation - you would need to parse the model's response
        # to identify which tools to use and their arguments
        try:
            # Execute the identified tool
            # For now, we'll just return the response
            return response
        except Exception as e:
            return f"Error executing task: {str(e)}"

    def add_tool(self, name: str, tool: Tool) -> None:
        """
        Add a new tool to the agent's toolbox.
        
        Args:
            name: The name of the tool
            tool: The tool implementation
        """
        self._toolbox[name] = tool

    def remove_tool(self, name: str) -> None:
        """
        Remove a tool from the agent's toolbox.
        
        Args:
            name: The name of the tool to remove
        """
        if name in self._toolbox:
            del self._toolbox[name] 