"""
Prompt loader utility for loading prompts from promptfoo configuration.
"""
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import pdb

class PromptLoader:
    """Loads prompts from promptfoo configuration files."""

    def __init__(self, promptfoo_dir: str = "./promptfoo", verbose=False):
        """Initialize with promptfoo directory path."""
        self.promptfoo_dir = Path(promptfoo_dir)
        if verbose:
            print("Initializing PromptLoader with directory:", self.promptfoo_dir)
        self.prompts_dir = self.promptfoo_dir / "prompts"
        self._prompt_cache = {}

    def load_prompt_by_name(self, name: str) -> Optional[str]:
        """
        Load a prompt by its name tag from promptfoo configuration.

        Args:
            name: The name tag of the prompt to load

        Returns:
            The prompt text with placeholders, or None if not found
        """

        if name in self._prompt_cache:
            return self._prompt_cache[name]

        # Search through all prompt files
        for prompt_file in self.promptfoo_dir.glob("*.yaml"):
            try:
                # print(prompt_file)
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)

                if prompt_data.get('name') == name:
                    if 'prompts' in prompt_data:  # Chat prompt
                        prompt_json_file = prompt_data.get('prompts', '')
                        if prompt_json_file:
                            try:
                                json_file_path = Path(self.promptfoo_dir / prompt_json_file[0].removeprefix('file://'))
                                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                                    prompt_dict = yaml.safe_load(json_file)
                                    prompt_dict = {
                                        "system": prompt_dict[0]["content"].replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}"),
                                        "user": prompt_dict[1]["content"].replace("{{", "{").replace("}}", "}").replace("{{", "{").replace("}}", "}")
                                    }
                                    self._prompt_cache[name] = prompt_dict
                                    return prompt_dict
                            except (FileNotFoundError, IndexError, KeyError) as e:
                                continue
                    else:  # Single prompt
                        prompt_text = prompt_data.get('prompt', '')
                        prompt_text = prompt_text.replace("{{", "{").replace("}}", "}")
                        prompt_dict = {"user": prompt_text}
                        self._prompt_cache[name] = prompt_dict
                        return prompt_dict

            except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
                print(f"Warning: Error loading prompt from {prompt_file}: {e}")
                continue

        print(f"Warning: Prompt with name '{name}' not found")
        return None

    def get_prompt_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a prompt by name.

        Args:
            name: The name tag of the prompt

        Returns:
            Metadata dictionary or None if not found
        """
        for prompt_file in self.promptfoo_dir.glob("*.yaml"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)

                if prompt_data.get('name') == name:
                    return prompt_data.get('metadata', {})

            except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
                continue

        return None

    def list_available_prompts(self) -> Dict[str, str]:
        """
        List all available prompts with their names and descriptions.

        Returns:
            Dictionary mapping prompt names to descriptions
        """
        prompts = {}

        for prompt_file in self.promptfoo_dir.glob("*.yaml"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)

                name = prompt_data.get('name')
                description = prompt_data.get('description', 'No description')

                if name:
                    prompts[name] = description

            except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
                continue

        return prompts

    def format_prompt(self, name: str, **kwargs) -> Optional[str]:
        """
        Load and format a prompt with the given variables.

        Args:
            name: The name tag of the prompt
            **kwargs: Variables to substitute in the prompt

        Returns:
            Formatted prompt text or None if not found
        """
        try:
            prompt_template = self.load_prompt_by_name(name)
            if not prompt_template:
                return None
            # If the prompt is a dictionary (e.g., chat with system and user roles)
            if isinstance(prompt_template, str):
                # Replace double curly braces with single curly braces
                prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")
                # Format the string template
                return prompt_template.format(**kwargs)
            elif isinstance(prompt_template, dict):
                retdict = {}
                # Format each template in the dictionary
                for role, template in prompt_template.items():
                    # Replace double curly braces with single curly braces
                    template = template.replace("{{", "{").replace("}}", "}")
                    retdict[role] = template.format(**kwargs)
                return retdict
            else:
                raise ValueError(f"Unsupported prompt template type: {type(prompt_template)}")
        except Exception as e:
            print(f"Error formatting prompt '{name}': {e}")
            return None
