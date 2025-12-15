#!/usr/bin/env python3
"""
LLM calling module with flexible prompt templating and batch processing.

Suppose we have 1000 headlines in a dataframe and we want to apply a prompt to each one.
Some stuff we might want
- structured output, like ideally apply prompts to this column and put results in a new column
- output validation, so llm doesn't e.g. transpose rows
- batching , don't send 1000 at once but don't send a single headline with a large prompt 1000 times
- concurrency / async processing, send many batches at once (but maybe specify some max concurrency)
- retry logic with exponential backoff

This module provides the LLMagent class for making structured LLM calls with:
- Flexible prompt templates with variable substitution
- Single and batch processing modes
- Retry logic with exponential backoff
- Pydantic output validation
- Async batch processing with concurrency control

todo: function that takes a dataframe , list of input columns, name of output column, and
"""

import asyncio
import json
import logging
import math
from typing import Any, Dict, List, Type, Optional, Tuple
from pydantic import BaseModel, ValidationError
import pandas as pd
import os

import openai
from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)

from config import DEFAULT_CONCURRENCY
from agents import Agent, Runner

import langfuse

_logger = logging.getLogger(__name__)

# Global singleton LangfuseClient for reuse across calls
_global_langfuse_client: Optional['LangfuseClient'] = None


def get_langfuse_client(logger: Optional[logging.Logger] = None) -> 'LangfuseClient':
    """
    Get or create singleton LangfuseClient.

    Args:
        logger: Optional logger instance

    Returns:
        Shared LangfuseClient instance
    """
    global _global_langfuse_client
    if _global_langfuse_client is None:
        _global_langfuse_client = LangfuseClient(logger=logger)
    return _global_langfuse_client


async def paginate_df_async(df: pd.DataFrame, chunk_size: int = 25):
    """Async generator for DataFrame pagination."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]
        await asyncio.sleep(0)  # Allow other tasks to run


async def paginate_list_async(lst, chunk_size: int = 25):
    """Async generator for list pagination."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
        await asyncio.sleep(0)  # Allow other tasks to run


def _introspect_output_type(output_type: Type[BaseModel]) -> tuple[Optional[str], Optional[str]]:
    """
    Introspect a Pydantic model to find list field and value field.

    Args:
        output_type: Pydantic BaseModel class

    Returns:
        Tuple of (item_list_field, value_field)
        - item_list_field: Name of the field containing List[X]
        - value_field: Name of the non-id field in the inner model X
    """
    import typing

    item_list_field = None
    value_field = None

    # Find the list field
    for field_name, field_info in output_type.model_fields.items():
        # Get the field type annotation
        field_type = field_info.annotation

        # Check if it's a List type
        origin = typing.get_origin(field_type)
        if origin is list or origin is List:
            item_list_field = field_name

            # Get the inner type of the list
            args = typing.get_args(field_type)
            if args and len(args) > 0:
                inner_type = args[0]

                # If inner type is a BaseModel, find the non-id field
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    inner_fields = inner_type.model_fields
                    non_id_fields = [name for name in inner_fields.keys()
                                     if name.lower() not in ('id', 'index')]

                    # If exactly one non-id field, use it as value_field
                    if len(non_id_fields) == 1:
                        value_field = non_id_fields[0]
            break

    return item_list_field, value_field


class LangfuseClient:
    """
    Client for retrieving prompts from Langfuse.

    Provides a clean interface to fetch prompts with system/user content and model configuration
    from Langfuse prompt management.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Langfuse client.

        Args:
            logger: Optional logger instance to use instead of module logger

        Raises:
            ImportError: If langfuse is not available
        """
        self.logger = logger or _logger
        self.client = langfuse.get_client()

        if self.logger:
            self.logger.info("Initialized LangfuseClient")

    @retry(
        retry=retry_if_exception_type((
            Exception,  # Catch all exceptions for Langfuse API calls
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    def get_prompt(self, prompt_name: str) -> tuple[str, str, str, str]:
        """
        Retrieve a prompt from Langfuse and extract system/user prompts and model configuration.

        Args:
            prompt_name: Name of the prompt in Langfuse (e.g., 'newsagent/headline_classifier')

        Returns:
            Tuple containing (system_prompt, user_prompt, model, reasoning_effort)

        Raises:
            ValueError: If prompt format is invalid or missing required content
            Exception: If prompt retrieval fails
        """
        try:
            if self.logger:
                self.logger.debug(
                    f"Attempting to retrieve prompt '{prompt_name}' from Langfuse")

            # Get prompt from Langfuse
            lf_prompt = self.client.get_prompt(prompt_name)

            if self.logger:
                self.logger.info(
                    f"Retrieved prompt '{prompt_name}' from Langfuse")

            # Validate prompt structure
            if not hasattr(lf_prompt, 'prompt') or not isinstance(lf_prompt.prompt, list):
                raise ValueError(
                    f"Invalid prompt format for '{prompt_name}': missing or invalid 'prompt' field")

            if len(lf_prompt.prompt) < 2:
                raise ValueError(
                    f"Invalid prompt format for '{prompt_name}': expected at least 2 prompt parts (system, user)")

            # Extract system and user prompts
            try:
                system_prompt = lf_prompt.prompt[0]['content']
                user_prompt = lf_prompt.prompt[1]['content']
            except (KeyError, IndexError) as e:
                raise ValueError(
                    f"Invalid prompt structure for '{prompt_name}': {e}")

            # Extract configuration
            config = lf_prompt.config if hasattr(lf_prompt, 'config') else {}
            model = config.get("model", "gpt-5")
            reasoning_effort = config.get("reasoning_effort", "medium")

            if self.logger:
                self.logger.info(
                    f"Parsed prompt '{prompt_name}': model={model}, reasoning_effort={reasoning_effort}, system_len={len(system_prompt)}, user_len={len(user_prompt)}")

            return (system_prompt, user_prompt, model, reasoning_effort)

        except Exception as e:
            error_msg = f"Failed to retrieve prompt '{prompt_name}': {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise Exception(error_msg) from e

    def create_llm_agent(self, prompt_name: str, output_type: Type[BaseModel],
                         verbose: bool = False, logger: Optional[logging.Logger] = None) -> 'LLMagent':
        """
        Convenience method to create an LLMagent from a Langfuse prompt.

        Args:
            prompt_name: Name of the prompt in Langfuse
            output_type: Pydantic model class for structured output
            verbose: Enable verbose logging
            logger: Optional logger instance

        Returns:
            Configured LLMagent instance
        """
        prompt_data = self.get_prompt(prompt_name)

        return LLMagent(
            system_prompt=prompt_data[0],
            user_prompt=prompt_data[1],
            output_type=output_type,
            model=prompt_data[2],
            reasoning_effort=prompt_data[3],
            verbose=verbose,
            logger=logger or self.logger
        )


class LLMagent(Agent):
    """
    General-purpose LLM agent for making structured calls with flexible prompt templating.

    Supports:
    - Multiple variable substitution in prompt templates
    - Single prompt calls with keyword arguments or dictionaries
    - Batch processing with async concurrency control
    - Retry logic with exponential backoff
    - Pydantic output validation
    """

    def __init__(self,
                 system_prompt: str,
                 user_prompt: str,
                 output_type: Type[BaseModel],
                 model: str,
                 verbose: bool = False,
                 logger: Optional[logging.Logger] = None,
                 reasoning_effort: Optional[str] = None,
                 trace_enable: Optional[bool] = None,
                 trace_tag_list: Optional[List[str]] = None):
        """
        Initialize the LLMagent

        Args:
            system_prompt: The system prompt template with variable placeholders (e.g., "You are a {role} assistant")
            user_prompt: The user prompt template with variable placeholders (e.g., "Analyze this {content_type}: {input}")
            output_type: Pydantic model class for structured output
            model: Model string (e.g., "gpt-4o")
            verbose: Enable verbose logging
            logger: Optional logger instance to use instead of module logger
            reasoning_effort: Optional reasoning effort level for reasoning models ("low", "medium", "high")
            trace_enable: Enable Langfuse tracing for this agent (overrides LANGFUSE_TRACING_ENABLED env var)
            trace_tag_list: List of tags for Langfuse traces (e.g., ["step_04_extract_summaries", "summary_agent"])
        """
        super().__init__(
            name="LLMagent",
            model=model,
            instructions=system_prompt,
            output_type=output_type
        )
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.output_type = output_type
        self.trace_tag = trace_tag_list
        self.verbose = verbose
        self.logger = logger or _logger

        # Validate and store reasoning_effort
        if reasoning_effort is not None:
            valid_efforts = {"low", "medium", "high"}
            if reasoning_effort not in valid_efforts:
                raise ValueError(
                    f"reasoning_effort must be one of {valid_efforts}, got: {reasoning_effort}"
                )
        self.reasoning_effort = reasoning_effort

        # Configure Langfuse tracing
        env_trace_enable = os.getenv(
            'LANGFUSE_TRACING_ENABLED', 'false').lower() == 'true'
        self.trace_enable = trace_enable if trace_enable is not None else env_trace_enable

        # Initialize appropriate OpenAI client
        if self.trace_enable:
            try:
                from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
                self.openai_client = LangfuseAsyncOpenAI()
                if self.verbose:
                    self.logger.info(
                        f"Initialized with Langfuse tracing enabled (tags: {self.trace_tag})")
            except ImportError:
                self.logger.warning(
                    "Langfuse tracing requested but langfuse.openai not available, using standard client")
                self.openai_client = AsyncOpenAI()
                self.trace_enable = False
        else:
            self.openai_client = AsyncOpenAI()

        if self.verbose:
            self.logger.info(f"""Initialized LLMagent:
system_prompt: {self.system_prompt}
user_prompt: {self.user_prompt}
output_type: {output_type.__name__}
model: {self.model}
trace_enable: {self.trace_enable}
trace_tag: {self.trace_tag}
schema: {json.dumps(output_type.model_json_schema(), indent=2)}
""")

    def _build_langfuse_metadata(self) -> Dict[str, Any]:
        """
        Build metadata dict for Langfuse tracing with tags.

        Returns:
            Dictionary with langfuse_tags if trace_tag is populated

        Note: When using langfuse.openai wrapper, tags should be passed as a list.
        The Langfuse wrapper handles this format specially.
        """
        metadata = {}
        if self.trace_tag:
            # Langfuse expects tags as a list/array
            metadata["langfuse_tags"] = self.trace_tag
        return metadata

    def _format_prompts(self, variables: Dict[str, Any]) -> str:
        """
        Format user prompt with variable substitution

        Args:
            variables: Dictionary of variables to substitute in user prompt template

        Returns:
            Formatted user prompt
        """
        try:
            formatted_user = self.user_prompt.format(**variables)
            return formatted_user
        except KeyError as e:
            raise ValueError(
                f"Missing required variable in prompt template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompts: {e}")

    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            # Retry on Pydantic validation errors (e.g., LLM returned wrong schema)
            ValidationError,
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict(self, variables: Dict[str, Any]) -> Any:
        """
        Make a single LLM call with dictionary-based variable substitution

        Args:
            variables: Dictionary of variables to substitute in prompt templates

        Returns:
            Single result of the specified output type
        """
        user_message = self._format_prompts(variables)
        user_message = user_message.strip()

        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        try:
            results = await Runner.run(self, user_message)
        except ValidationError as e:
            # Log detailed information about validation failure
            self.logger.error(
                f"Pydantic validation error for {self.output_type.__name__}: {e}"
            )
            self.logger.error(
                f"Expected schema: {self.output_type.model_json_schema()}"
            )
            # Re-raise to trigger tenacity retry
            raise

        if self.verbose:
            self.logger.info(f"Result: {results}")

        return results.final_output if hasattr(results, 'final_output') else results

    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            # Retry on Pydantic validation errors (e.g., LLM returned wrong schema)
            ValidationError,
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict_chat(self, variables: Dict[str, Any], reasoning_effort: Optional[str] = None) -> Any:
        """
        Make a single LLM call using OpenAI chat completions API directly

        Args:
            variables: Dictionary of variables to substitute in prompt templates
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
                            Only applies to reasoning models (o1, o3-mini, gpt-5).
                            Overrides instance-level reasoning_effort if provided.

        Returns:
            Single result of the specified output type
        """
        user_message = self._format_prompts(variables)
        user_message = user_message.strip()

        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        # Determine reasoning_effort to use (parameter overrides instance attribute)
        effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort

        # Check if reasoning_effort is requested but not supported
        if effective_reasoning_effort is not None and not self._supports_reasoning_effort():
            self.logger.warning(
                f"reasoning_effort='{effective_reasoning_effort}' specified but model '{self.model}' "
                f"does not support this parameter. It will be ignored."
            )
            effective_reasoning_effort = None

        # Use instance OpenAI client (Langfuse-wrapped if tracing enabled)
        client = self.openai_client

        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Note: Langfuse tracing is handled automatically by langfuse.openai.AsyncOpenAI wrapper
        # The wrapper captures model, messages, and response data without needing explicit metadata updates

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "schema": self.output_type.model_json_schema()
                    }
                },
                "safety_identifier": "news_agent"
            }

            # Only add metadata when Langfuse tracing is enabled
            # (standard OpenAI API doesn't accept metadata without store=True)
            if self.trace_enable:
                api_params["metadata"] = self._build_langfuse_metadata()

            # Add reasoning_effort if supported
            if effective_reasoning_effort is not None:
                api_params["reasoning_effort"] = effective_reasoning_effort
                if self.verbose:
                    self.logger.info(
                        f"Using reasoning_effort: {effective_reasoning_effort}")

            # Make the chat completion call with structured output
            response = await client.chat.completions.create(**api_params)

            # Check for refusal
            message = response.choices[0].message
            if hasattr(message, 'refusal') and message.refusal:
                self.logger.error(
                    f"LLM refused request. User message: {user_message}")
                self.logger.error(f"Refusal reason: {message.refusal}")
                raise ValueError(f"LLM refused the request: {message.refusal}")

            # Parse the JSON response into the Pydantic model
            response_text = message.content
            response_json = json.loads(response_text)

            try:
                result = self.output_type.model_validate(response_json)
            except ValidationError as e:
                # Log detailed information about validation failure
                self.logger.error(
                    f"Pydantic validation error for {self.output_type.__name__}: {e}"
                )
                self.logger.error(f"Response JSON: {response_json}")
                self.logger.error(
                    f"Expected schema: {self.output_type.model_json_schema()}"
                )
                # Re-raise to trigger tenacity retry
                raise

        except BadRequestError as e:
            self.logger.error(f"BadRequestError: {e}")
            self.logger.error(
                f"User message that caused error: {user_message}")
            raise

        if self.verbose:
            self.logger.info(f"Result: {result}")

        return result

    def _supports_logprobs(self) -> bool:
        """
        Check if the current model supports logprobs functionality.

        Returns:
            bool: True if model supports logprobs, False otherwise
        """
        # Models that support logprobs
        logprobs_supported_models = {
            "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano",
            "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
        }

        # Check exact match or partial match for versioned models
        model_name = self.model.lower()
        if model_name in logprobs_supported_models:
            return True

        # Check for partial matches (e.g., gpt-4.1-mini-2024-07-18)
        for supported_model in logprobs_supported_models:
            if model_name.startswith(supported_model):
                return True

        return False

    def _supports_reasoning_effort(self) -> bool:
        """
        Check if the current model supports reasoning_effort parameter.

        Returns:
            bool: True if model supports reasoning_effort, False otherwise
        """
        # Models that support reasoning_effort (reasoning models only)
        reasoning_models = {
            "o1", "o1-preview", "o3-mini", "gpt-5"
        }

        # Check exact match or partial match for versioned models
        model_name = self.model.lower()
        if model_name in reasoning_models:
            return True

        # Check for partial matches (e.g., o3-mini-2025-01-31)
        for reasoning_model in reasoning_models:
            if model_name.startswith(reasoning_model):
                return True

        return False

    def _extract_token_probabilities(self, logprobs_data: Dict, target_tokens: List[str]) -> Dict[str, float]:
        """
        Extract probabilities for specific target tokens from OpenAI logprobs response.

        Args:
            logprobs_data: Raw logprobs data from OpenAI API response
            target_tokens: List of tokens to extract probabilities for (e.g., ["1", "0"])

        Returns:
            Dict mapping token to probability (e.g., {"1": 0.85, "0": 0.15})
        """
        if not logprobs_data or getattr(logprobs_data, 'content', None) is None:
            raise ValueError(
                "Invalid logprobs_data. Must contain 'content' key with non-None value.")

        # Look at the first token's logprobs (for binary classification, answer should be first token)
        first_token_logprobs = logprobs_data.content[0]

        if not hasattr(first_token_logprobs, 'top_logprobs'):
            raise ValueError(
                "Invalid first_token_logprobs. Could not find 'top_logprobs' key or 'top_logprobs' is empty."
            )

        # Extract probabilities for target tokens
        result = {}
        top_logprobs = first_token_logprobs.top_logprobs

        for target_token in target_tokens:
            # Find matching token in top_logprobs
            found_prob = 0.0
            for token_info in top_logprobs:
                if token_info.token == target_token:
                    # Convert log probability to probability: p = e^(logprob)
                    found_prob = math.exp(token_info.logprob)
                    break
            result[target_token] = found_prob

        return result

    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict_chat_probs(self, variables: Dict[str, Any], top_logprobs: int = 5) -> Tuple[str, Dict]:
        """
        Make a single LLM call with logprobs enabled (no structured output).

        This method skips structured output (json_schema) to enable logprobs functionality,
        since OpenAI's structured outputs are incompatible with logprobs.

        Args:
            variables: Dictionary of variables to substitute in prompt templates
            top_logprobs: Number of top tokens to return probabilities for (0-5)

        Returns:
            Tuple of (response_text, logprobs_data)

        Raises:
            ValueError: If model doesn't support logprobs
        """
        if not self._supports_logprobs():
            supported_models = ["gpt-4.1-mini", "gpt-4o-mini",
                                "gpt-4.1", "gpt-4o", "gpt-4-turbo"]
            raise ValueError(
                f"Model '{self.model}' does not support logprobs. "
                f"Supported models: {supported_models}"
            )

        user_message = self._format_prompts(variables)
        user_message = user_message.strip()

        if self.verbose:
            self.logger.info(f"User message (with logprobs): {user_message}")

        # Use instance OpenAI client (Langfuse-wrapped if tracing enabled)
        client = self.openai_client

        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Note: Langfuse tracing is handled automatically by langfuse.openai.AsyncOpenAI wrapper
        # The wrapper captures model, messages, logprobs, and response data automatically

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": top_logprobs,
                "safety_identifier": "news_agent"
            }

            # Only add metadata when Langfuse tracing is enabled
            # (standard OpenAI API doesn't accept metadata without store=True)
            if self.trace_enable:
                api_params["metadata"] = self._build_langfuse_metadata()

            # Make the chat completion call with logprobs (NO structured output)
            response = await client.chat.completions.create(**api_params)

            # Check for refusal
            message = response.choices[0].message
            if hasattr(message, 'refusal') and message.refusal:
                self.logger.error(
                    f"LLM refused request. User message: {user_message}")
                self.logger.error(f"Refusal reason: {message.refusal}")
                raise ValueError(f"LLM refused the request: {message.refusal}")

            # Extract response text and logprobs
            response_text = message.content
            logprobs_data = response.choices[0].logprobs

        except BadRequestError as e:
            self.logger.error(f"BadRequestError: {e}")
            self.logger.error(
                f"User message that caused error: {user_message}")
            raise

        if self.verbose:
            self.logger.info(f"Response text: {response_text}")
            self.logger.info(
                f"Logprobs available: {logprobs_data is not None}")

        return response_text, logprobs_data

    async def run_prompt(self, reasoning_effort: Optional[str] = None, **kwargs) -> Any:
        """
        Make a single LLM call with keyword argument variable substitution
        (don't use prompt to name it since the base class has a prompt method)

        Args:
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
                            Only applies to reasoning models (o1, o3-mini, gpt-5).
                            Overrides instance-level reasoning_effort if provided.
            **kwargs: Keyword arguments to substitute in prompt templates

        Returns:
            Single result of the specified output type
        """
        # Repackage kwargs into a dictionary and call prompt_dict_chat
        return await self.prompt_dict_chat(kwargs, reasoning_effort=reasoning_effort)

    async def run_prompt_with_probs(self, target_tokens: List[str] = ["1"], **kwargs) -> Dict[str, float]:
        """
        Make a single LLM call and return probabilities for specific target tokens.

        This method is designed for binary classification tasks where you want to get
        the probability of specific tokens (e.g., "1" for spam classification).

        Args:
            target_tokens: List of tokens to get probabilities for (default: ["1"])
            **kwargs: Keyword arguments to substitute in prompt templates

        Returns:
            Dict mapping each target token to its probability (e.g., {"1": 0.85})

        Example:
            # For spam classification
            agent = LLMagent(
                system_prompt="Classify as spam or not spam",
                user_prompt="Text: {text}\\nReturn only 1 for spam, 0 for not spam",
                output_type=str,  # Not used for logprobs
                model="gpt-4.1-mini"
            )

            probs = await agent.run_prompt_with_probs(
                target_tokens=["1"],
                text="Buy now limited time offer!!!"
            )
            spam_probability = probs["1"]  # e.g., 0.89
        """
        # Get raw response and logprobs
        response_text, logprobs_data = await self.prompt_dict_chat_probs(kwargs)

        # Extract probabilities for target tokens
        probabilities = self._extract_token_probabilities(
            logprobs_data, target_tokens)

        if self.verbose:
            self.logger.info(f"Token probabilities: {probabilities}")

        return probabilities

    async def prompt_batch(self,
                           variables_list: List[Dict[str, Any]],
                           batch_size: int = 25,
                           max_concurrency: int = DEFAULT_CONCURRENCY,
                           retries: int = 3,
                           item_list_field: str = 'results_list',
                           item_id_field: str = '',
                           chat: bool = True) -> List[Any]:
        """
        Process a list of variable dictionaries using true batch calls.

        Note: This method assumes the prompt template expects a single 'input_str' parameter.
        All items from each batch will be converted to string and processed in a single API call
        per batch, dramatically reducing cost and improving performance.

        Args:
            variables_list: List of variable dictionaries for prompt substitution
            batch_size: Number of items to process in each batch
            max_concurrency: Maximum number of concurrent requests
            retries: Number of retry attempts for failed requests
            item_id_field: Optional ID field name for validation. If provided, validates that each sent ID matches a received ID
            chat: If True (default), use prompt_dict_chat; if False, use prompt_dict

        Returns:
            List of results maintaining original input order
        """
        if not variables_list:
            return []

        # Split into batches
        batches = [variables_list[i:i+batch_size]
                   for i in range(0, len(variables_list), batch_size)]

        sem = asyncio.Semaphore(max_concurrency)
        if self.verbose:
            self.logger.info(
                f"Processing {len(batches)} batches with concurrency {max_concurrency}")

        async def _process_batch(batch_idx: int, batch_variables: List[Dict[str, Any]]) -> tuple[int, List[Any]]:
            """Process a single batch with retry logic"""
            last_exc = None

            for attempt in range(retries):
                try:
                    async with sem:
                        # Process the entire batch in a single API call
                        if chat:
                            result = await self.prompt_dict_chat({'input_str': str(batch_variables)})
                        else:
                            result = await self.prompt_dict({'input_str': str(batch_variables)})
                        batch_results = result

                        # Validate IDs if item_id_field is specified
                        if item_id_field:
                            sent_ids = [var.get(item_id_field)
                                        for var in batch_variables]
                            received_ids = []

                            for result in batch_results:
                                if hasattr(result, item_id_field):
                                    received_ids.append(
                                        getattr(result, item_id_field))
                                elif isinstance(result, dict) and item_id_field in result:
                                    received_ids.append(result[item_id_field])
                                else:
                                    raise ValueError(
                                        f"Result missing required ID field '{item_id_field}': {result}")

                            # Check if all sent IDs have corresponding received IDs
                            sent_set = set(sent_ids)
                            received_set = set(received_ids)

                            if sent_set != received_set:
                                missing_ids = sent_set - received_set
                                extra_ids = received_set - sent_set
                                error_msg = f"ID mismatch in batch {batch_idx}:"
                                if missing_ids:
                                    error_msg += f" Missing IDs: {missing_ids}"
                                if extra_ids:
                                    error_msg += f" Extra IDs: {extra_ids}"
                                raise ValueError(error_msg)

                        return batch_idx, batch_results

                except Exception as e:
                    last_exc = e
                    self.logger.warning(
                        f"Batch {batch_idx} attempt {attempt + 1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)

            # If all retries failed, raise the last exception
            raise last_exc or RuntimeError(
                f"Unknown error processing batch {batch_idx}")

        # Create tasks for all batches
        tasks = [
            asyncio.create_task(_process_batch(i, batch))
            for i, batch in enumerate(batches)
        ]

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)

        if item_list_field:
            # Reassemble results in original order
            flattened_results = []
            flattened_success = False
            for batch_idx, results in sorted(batch_results, key=lambda x: x[0]):
                if hasattr(results, item_list_field):
                    flattened_results.extend(getattr(results, item_list_field))
                else:
                    break
                flattened_success = True

            if flattened_success:
                # Validate final result count
                if len(flattened_results) != len(variables_list):
                    raise ValueError(
                        f"Result count mismatch: expected {len(variables_list)}, got {len(flattened_results)}"
                    )
                else:
                    return flattened_results
            else:  # return unflattened results
                return batch_results
        else:  # return unflattened results
            return batch_results

    async def filter_dataframe_chunk(self,
                                     input_df: pd.DataFrame,
                                     input_vars: Optional[Dict[str,
                                                               Any]] = None,
                                     item_list_field: str = 'results_list',
                                     item_id_field: str = 'id',
                                     retries: int = 3,
                                     chat: bool = True
                                     ) -> Any:
        """
        Process a single DataFrame asynchronously using Agent SDK.

        Applies the configured system and user prompts to a DataFrame converted to JSON,
        with configurable delimiters and additional input variables.

        Note: This method expects the user_prompt template to contain an {input_text} placeholder
        where the DataFrame JSON will be substituted.

        Args:
            input_df: The DataFrame to process
            input_vars: Optional additional variables for prompt substitution
            item_list_field: Name of the field in the response that contains the list of results
            item_id_field: Name of the ID field to validate matches between sent and received data
            retries: Number of retry attempts for validation failures
            chat: If True (default), use prompt_dict_chat; if False, use prompt_dict

        Returns:
            Single result of the configured output_type (structured Pydantic object)
        """
        expected_count = len(input_df)
        last_exc = None

        for attempt in range(retries):
            try:
                # Convert DataFrame to JSON
                input_text = input_df.to_json(orient='records', indent=2)

                # Prepare the input dictionary
                input_dict = {"input_text": input_text}
                # add input_vars if provided
                if input_vars is not None:
                    input_dict.update(input_vars)
                # Use prompt_dict_chat or prompt_dict based on chat parameter
                if chat:
                    result = await self.prompt_dict_chat(input_dict)
                else:
                    result = await self.prompt_dict(input_dict)
                # Validate item count and IDs if item_list_field is specified
                if item_list_field:
                    if hasattr(result, item_list_field):
                        result_list = getattr(result, item_list_field)
                        if isinstance(result_list, list):
                            received_count = len(result_list)
                            if received_count != expected_count:
                                error_msg = f"Item count mismatch: expected {expected_count}, got {received_count}"
                                self.logger.warning(
                                    f"Attempt {attempt + 1}/{retries}: {error_msg}")
                                if attempt < retries - 1:
                                    # Exponential backoff
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                else:
                                    raise ValueError(error_msg)

                            # Validate IDs if item_id_field is specified and exists in DataFrame
                            if item_id_field and item_id_field in input_df.columns:
                                sent_ids = input_df[item_id_field].tolist()
                                received_ids = []

                                for item in result_list:
                                    if hasattr(item, item_id_field):
                                        received_ids.append(
                                            getattr(item, item_id_field))
                                    elif isinstance(item, dict) and item_id_field in item:
                                        received_ids.append(
                                            item[item_id_field])
                                    else:
                                        error_msg = f"Result item missing required ID field '{item_id_field}': {item}"
                                        self.logger.warning(
                                            f"Attempt {attempt + 1}/{retries}: {error_msg}")
                                        if attempt < retries - 1:
                                            await asyncio.sleep(2 ** attempt)
                                            continue
                                        else:
                                            raise ValueError(error_msg)

                                # Validate ID order - sent and received must match exactly (order + presence)
                                if sent_ids != received_ids:
                                    # Provide detailed error information
                                    sent_set = set(sent_ids)
                                    received_set = set(received_ids)
                                    missing_ids = sent_set - received_set
                                    extra_ids = received_set - sent_set

                                    if missing_ids or extra_ids:
                                        error_msg = "ID presence mismatch:"
                                        if missing_ids:
                                            error_msg += f" Missing IDs: {missing_ids}"
                                        if extra_ids:
                                            error_msg += f" Extra IDs: {extra_ids}"
                                    else:
                                        error_msg = f"ID order mismatch: sent {sent_ids} != received {received_ids}"

                                    self.logger.warning(
                                        f"Attempt {attempt + 1}/{retries}: {error_msg}")
                                    if attempt < retries - 1:
                                        # Exponential backoff
                                        await asyncio.sleep(2 ** attempt)
                                        continue
                                    else:
                                        raise ValueError(error_msg)

                        else:
                            raise ValueError(
                                f"Field '{item_list_field}' is not a list: {type(result_list)}")
                    else:
                        raise ValueError(
                            f"Result missing required field '{item_list_field}': {result}")

                return result

            except asyncio.TimeoutError as e:
                last_exc = e
                self.logger.error(
                    f"Timeout error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except (ConnectionError, TimeoutError) as e:
                last_exc = e
                self.logger.error(
                    f"Network/timeout error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except ValueError as e:
                last_exc = e
                self.logger.error(
                    f"Invalid data in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                last_exc = e
                self.logger.error(
                    f"Unexpected error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        # If we get here, all retries failed
        raise last_exc or RuntimeError(
            f"Unknown error after {retries} attempts")

    async def _process_indexed_chunk(self,
                                     chunk_idx: int,
                                     chunk_df: pd.DataFrame,
                                     sem: asyncio.Semaphore,
                                     input_vars: Optional[Dict[str,
                                                               Any]] = None,
                                     item_list_field: str = 'results_list',
                                     item_id_field: str = 'id',
                                     retries: int = 3,
                                     chat: bool = True) -> tuple[int, Any]:
        """
        Process a single chunk and return with its index for order preservation.

        Args:
            chunk_idx: Index of this chunk in the original chunk list
            chunk_df: DataFrame chunk to process
            sem: Semaphore for concurrency control
            input_vars: Optional additional variables for prompt substitution
            item_list_field: Name of the field in response containing results list
            item_id_field: Name of the ID field to validate matches
            retries: Number of retry attempts for validation failures
            chat: If True (default), use prompt_dict_chat; if False, use prompt_dict

        Returns:
            Tuple of (chunk_index, result) for order preservation
        """
        async with sem:
            result = await self.filter_dataframe_chunk(
                chunk_df,
                input_vars=input_vars,
                item_list_field=item_list_field,
                item_id_field=item_id_field,
                retries=retries,
                chat=chat
            )
            return chunk_idx, result

    async def filter_dataframe_batch(self,
                                     input_df: pd.DataFrame,
                                     input_vars: Optional[Dict[str,
                                                               Any]] = None,
                                     item_list_field: str = 'results_list',
                                     item_id_field: str = 'id',
                                     retries: int = 3,
                                     chunk_size: int = 25,
                                     return_series: bool = False,
                                     value_field: str = 'output',
                                     chat: bool = True,
                                     return_probabilities: bool = False,
                                     target_tokens: List[str] = None,
                                     max_concurrency: int = DEFAULT_CONCURRENCY,
                                     **kwargs
                                     ) -> Any:
        """
        Process a DataFrame in chunks asynchronously using concurrent calls to filter_dataframe_chunk.

        Chunks the input DataFrame using paginate_df_async and processes each chunk
        simultaneously with filter_dataframe_chunk. Chunks are processed with index tracking to
        guarantee correct ordering regardless of async completion timing.

        Args:
            input_df: The DataFrame to process
            input_vars: Optional additional variables for prompt substitution
            item_list_field: Name of the field in the response that contains the list of results
            item_id_field: Name of the ID field to validate matches between sent and received data
            retries: Number of retry attempts for validation failures per chunk
            chunk_size: Number of rows per chunk (default: 25)
            return_series: If True, return pandas Series for direct DataFrame assignment
            value_field: Field name to extract values from when return_series=True
            chat: If True (default), use prompt_dict_chat; if False, use prompt_dict
            return_probabilities: If True, return token probabilities instead of structured output
            target_tokens: List of tokens to extract probabilities for (default: ["1"])
            max_concurrency: Maximum number of concurrent chunk processing tasks (default: DEFAULT_CONCURRENCY)

        Returns:
            If return_probabilities=True: pandas Series with probabilities for target tokens
            If return_series=True: pandas Series with values for DataFrame column assignment
            Otherwise: Single concatenated result object (if item_list_field specified) or list of results
        """
        # print("concurrency: ", max_concurrency)
        if input_df.empty:
            return []

        # Use semaphore for concurrency control
        sem = asyncio.Semaphore(max_concurrency)

        # Handle probability extraction mode
        if return_probabilities:
            if target_tokens is None:
                target_tokens = ["1"]

            if not self._supports_logprobs():
                raise ValueError(
                    f"Model '{self.model}' does not support logprobs required for probability extraction"
                )

            # For probabilities, we process each row individually using run_prompt_with_probs

            async def _process_row_with_sem(row):
                async with sem:
                    # Convert row to dict and merge with input_vars
                    row_vars = row.to_dict()
                    if input_vars:
                        row_vars.update(input_vars)
                    # Get probabilities for this row
                    return await self.run_prompt_with_probs(target_tokens=target_tokens, **row_vars)

            # Process all rows asynchronously with concurrency control
            tasks = [_process_row_with_sem(row)
                     for _, row in input_df.iterrows()]
            prob_dicts = await asyncio.gather(*tasks)

            # Extract probability for first target token from each dict
            probabilities = [prob_dict.get(
                target_tokens[0], 0.0) for prob_dict in prob_dicts]

            # Return as Series indexed to match input DataFrame
            return pd.Series(probabilities, index=input_df.index)

        # Create chunks using the async generator
        chunks = []
        async for chunk in paginate_df_async(input_df, chunk_size):
            chunks.append(chunk)

        if not chunks:
            return []

        if self.verbose:
            self.logger.info(
                f"Processing {len(chunks)} chunks with concurrency {max_concurrency}")

        # Process all chunks concurrently with index tracking and semaphore control
        tasks = [
            self._process_indexed_chunk(
                i, chunk, sem,
                input_vars=input_vars,
                item_list_field=item_list_field,
                item_id_field=item_id_field,
                retries=retries,
                chat=chat,
                **kwargs
            )
            for i, chunk in enumerate(chunks)
        ]

        try:
            indexed_results = await asyncio.gather(*tasks)
            # Sort by chunk index to guarantee order
            sorted_results = sorted(indexed_results, key=lambda x: x[0])
        except Exception as e:
            self.logger.error(f"Error in filter_dataframe_batch: {e}")
            raise

        # If item_list_field is specified, concatenate all result lists in order
        if item_list_field:
            try:
                # Extract results in correct chunk order
                all_items = []
                for chunk_idx, chunk_result in sorted_results:
                    if hasattr(chunk_result, item_list_field):
                        result_list = getattr(chunk_result, item_list_field)
                        if isinstance(result_list, list):
                            all_items.extend(result_list)
                        else:
                            self.logger.error(
                                f"Field '{item_list_field}' is not a list: {type(result_list)}")
                            # Fall back to returning raw results
                            return [result for _, result in sorted_results]
                    else:
                        self.logger.error(
                            f"Result missing field '{item_list_field}': {chunk_result}")
                        # Fall back to returning raw results
                        return [result for _, result in sorted_results]

                # Check if we should return Series for DataFrame assignment
                if return_series:
                    values = [getattr(item, value_field) for item in all_items]
                    return pd.Series(values, index=input_df.index)

                # Create a new result object with concatenated items
                # Use the structure of the first result as template
                if sorted_results and hasattr(sorted_results[0][1], item_list_field):
                    first_result = sorted_results[0][1]
                    # Create a copy of the first result and replace the list field
                    concatenated_result = first_result.__class__(**{
                        **{k: v for k, v in first_result.__dict__.items() if k != item_list_field},
                        item_list_field: all_items
                    })
                    return concatenated_result
                else:
                    # If we can't create proper structure, return the items directly
                    return all_items

            except Exception as e:
                self.logger.error(f"Error concatenating results: {e}")
                # Fall back to returning raw results
                return [result for _, result in sorted_results]
        else:
            # No item_list_field specified, return list of results
            if return_series:
                self.logger.warning(
                    "return_series=True but no item_list_field specified, returning raw results")
            return [result for _, result in sorted_results]

    async def filter_dataframe(self, input_df: pd.DataFrame,
                               value_field: str = 'output',
                               **kwargs) -> pd.Series:
        """
        Process DataFrame and return values as Series for direct column assignment.

        This is a convenience method that wraps filter_dataframe_batch and extracts
        the specified field values as a pandas Series for direct DataFrame assignment.
        All chunk ordering and ID validation guarantees from filter_dataframe_batch apply.

        Args:
            input_df: DataFrame to process
            value_field: Field name to extract from results (default: 'output')
            **kwargs: All other arguments passed to filter_dataframe_batch
                     (item_list_field, item_id_field, retries, chunk_size, chat, etc.)

        Returns:
            pandas Series with extracted values, indexed to match input_df

        Examples:
            # Basic classification
            df["ai_related"] = await agent.filter_dataframe(df[['headline']])

            # Extract different field
            df["confidence"] = await agent.filter_dataframe(
                df[['text']],
                value_field='confidence'
            )

            # Get probabilities for binary classification
            df["spam_probability"] = await agent.filter_dataframe(
                df[['text']],
                return_probabilities=True,
                target_tokens=["1"]
            )

            # With ID validation
            df["sentiment"] = await agent.filter_dataframe(
                df[['text', 'id']],
                item_id_field='id',
                value_field='sentiment'
            )
        """
        # Call filter_dataframe_batch with all provided arguments
        result = await self.filter_dataframe_batch(input_df, **kwargs)

        # If result is already a Series (from probability extraction), return it directly
        if isinstance(result, pd.Series):
            return result

        # Extract values from the structured result
        if hasattr(result, 'results_list'):
            # Standard case: structured object with results_list
            values = [getattr(item, value_field)
                      for item in result.results_list]
        elif isinstance(result, list):
            # Fallback case: result is already a list of items
            values = [getattr(item, value_field) for item in result]
        else:
            # Unexpected result format
            raise ValueError(
                f"Unexpected result format from filter_dataframe_batch: {type(result)}")

        # Validate that we have the right number of values
        if len(values) != len(input_df):
            raise ValueError(
                f"Value count mismatch: expected {len(input_df)}, got {len(values)}")

        return pd.Series(values, index=input_df.index)


async def run_prompt_on_dataframe(
    input_df: pd.DataFrame,
    prompt_name: str,
    output_type: Type[BaseModel],
    value_field: Optional[str] = None,
    item_list_field: Optional[str] = None,
    item_id_field: str = 'id',
    chunk_size: int = 25,
    max_concurrency: int = DEFAULT_CONCURRENCY,
    return_probabilities: bool = False,
    target_tokens: Optional[List[str]] = None,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> pd.Series:
    """
    Convenience function to run a Langfuse prompt on a DataFrame.

    Automatically:
    - Fetches prompt from Langfuse using singleton client
    - Creates LLMagent with prompt configuration
    - Introspects output_type to determine field names (if not provided)
    - Runs inference and returns pandas Series

    Args:
        input_df: DataFrame to process
        prompt_name: Name of prompt in Langfuse (e.g., 'newsagent/rate_quality')
        output_type: Pydantic model class for structured output
        value_field: Field to extract from results (auto-detected if None)
        item_list_field: List field name in output (auto-detected if None)
        item_id_field: ID field name for validation (default: 'id')
        chunk_size: Rows per batch (default: 25)
        max_concurrency: Max concurrent requests (default: DEFAULT_CONCURRENCY)
        return_probabilities: Return token probabilities instead of structured output
        target_tokens: Tokens to extract probabilities for (default: ["1"])
        verbose: Enable verbose logging
        logger: Optional logger instance
        **kwargs: Additional arguments passed to filter_dataframe

    Returns:
        pandas Series with results, indexed to match input_df

    Example:
        # Single line replaces 10+ lines of boilerplate
        rating_df['low_quality'] = await run_prompt_on_dataframe(
            rating_df[['id', 'input_text']],
            "newsagent/rate_quality",
            StoryRatings,
            return_probabilities=True,
            logger=logger
        )
    """
    logger = logger or _logger

    # Get singleton Langfuse client
    lf_client = get_langfuse_client(logger=logger)

    # Fetch prompt from Langfuse
    system_prompt, user_prompt, model, reasoning_effort = lf_client.get_prompt(
        prompt_name)

    # Auto-detect fields if not provided
    if item_list_field is None or value_field is None:
        detected_list_field, detected_value_field = _introspect_output_type(
            output_type)

        if item_list_field is None:
            item_list_field = detected_list_field or 'results_list'
        if value_field is None:
            value_field = detected_value_field

            # If still None and not using probabilities, raise error
            if value_field is None and not return_probabilities:
                raise ValueError(
                    f"Could not auto-detect value_field for {output_type.__name__}. "
                    f"Please specify explicitly."
                )

    # Create LLM agent
    agent = LLMagent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=output_type,
        model=model,
        reasoning_effort=reasoning_effort,
        verbose=verbose,
        logger=logger
    )

    # Run inference
    result_series = await agent.filter_dataframe(
        input_df,
        value_field=value_field,
        item_list_field=item_list_field,
        item_id_field=item_id_field,
        chunk_size=chunk_size,
        max_concurrency=max_concurrency,
        return_probabilities=return_probabilities,
        target_tokens=target_tokens,
        **kwargs
    )

    return result_series
