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
from typing import Any, Dict, List, Type, Union, Optional
from pydantic import BaseModel
import pandas as pd

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

from agents import Agent, Runner
from log_handler import sanitize_error_for_logging

import langfuse

_logger = logging.getLogger(__name__)

async def paginate_df_async(df: pd.DataFrame, chunk_size: int = 25):
      """Async generator for DataFrame pagination."""
      for i in range(0, len(df), chunk_size):
          yield df.iloc[i:i + chunk_size]
          await asyncio.sleep(0)  # Allow other tasks to run


async def paginate_list_async(l, chunk_size: int = 25):
    """Async generator for list pagination."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]
        await asyncio.sleep(0)  # Allow other tasks to run

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

    def get_prompt(self, prompt_name: str) -> tuple[str, str, str]:
        """
        Retrieve a prompt from Langfuse and extract system/user prompts and model configuration.

        Args:
            prompt_name: Name of the prompt in Langfuse (e.g., 'newsagent/headline_classifier')

        Returns:
            Tuple containing (system_prompt, user_prompt, model)

        Raises:
            ValueError: If prompt format is invalid or missing required content
            Exception: If prompt retrieval fails
        """
        try:
            # Get prompt from Langfuse
            lf_prompt = self.client.get_prompt(prompt_name)

            if self.logger:
                self.logger.info(f"Retrieved prompt '{prompt_name}' from Langfuse")

            # Validate prompt structure
            if not hasattr(lf_prompt, 'prompt') or not isinstance(lf_prompt.prompt, list):
                raise ValueError(f"Invalid prompt format for '{prompt_name}': missing or invalid 'prompt' field")

            if len(lf_prompt.prompt) < 2:
                raise ValueError(f"Invalid prompt format for '{prompt_name}': expected at least 2 prompt parts (system, user)")

            # Extract system and user prompts
            try:
                system_prompt = lf_prompt.prompt[0]['content']
                user_prompt = lf_prompt.prompt[1]['content']
            except (KeyError, IndexError) as e:
                raise ValueError(f"Invalid prompt structure for '{prompt_name}': {e}")

            # Extract configuration
            config = lf_prompt.config if hasattr(lf_prompt, 'config') else {}
            model = config.get("model", "gpt-5")

            if self.logger:
                self.logger.info(f"Parsed prompt '{prompt_name}': model={model}, system_len={len(system_prompt)}, user_len={len(user_prompt)}")

            return (system_prompt, user_prompt, model)

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
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the LLMagent

        Args:
            system_prompt: The system prompt template with variable placeholders (e.g., "You are a {role} assistant")
            user_prompt: The user prompt template with variable placeholders (e.g., "Analyze this {content_type}: {input}")
            output_type: Pydantic model class for structured output
            model: Model string (e.g., "gpt-4o")
            verbose: Enable verbose logging
            logger: Optional logger instance to use instead of module logger
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
        self.verbose = verbose
        self.logger = logger or _logger

        if self.verbose:
            self.logger.info(f"""Initialized LLMagent:
system_prompt: {self.system_prompt}
user_prompt: {self.user_prompt}
output_type: {output_type.__name__}
model: {self.model}
schema: {json.dumps(output_type.model_json_schema(), indent=2)}
""")

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
            raise ValueError(f"Missing required variable in prompt template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompts: {e}")

    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def prompt_dict(self, variables: Dict[str, Any]) -> Any:
        """
        Make a single LLM call with dictionary-based variable substitution

        Args:
            variables: Dictionary of variables to substitute in prompt templates

        Returns:
            Single result of the specified output type
        """
        print(variables)
        user_message = self._format_prompts(variables)
        print(user_message)

        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        results = await Runner.run(self, user_message)

        if self.verbose:
            self.logger.info(f"Result: {results}")

        return results.final_output if hasattr(results, 'final_output') else results

    @retry(
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def run_prompt(self, **kwargs) -> Any:
        """
        Make a single LLM call with keyword argument variable substitution
        (don't use prompt to name it since the base class has a prompt method)
        Args:
            **kwargs: Keyword arguments to substitute in prompt templates

        Returns:
            Single result of the specified output type
        """
        user_message = self._format_prompts(kwargs)

        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        results = await Runner.run(self, user_message)

        if self.verbose:
            self.logger.info(f"Result: {results}")

        return results.final_output if hasattr(results, 'final_output') else results


    async def prompt_batch(self,
                          variables_list: List[Dict[str, Any]],
                          batch_size: int = 25,
                          max_concurrency: int = 16,
                          retries: int = 3,
                          item_list_field: str = 'results_list',
                          item_id_field: str = '') -> List[Any]:
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

        Returns:
            List of results maintaining original input order
        """
        if not variables_list:
            return []

        # Split into batches
        batches = [variables_list[i:i+batch_size]
                  for i in range(0, len(variables_list), batch_size)]

        sem = asyncio.Semaphore(max_concurrency)
        self.logger.info(f"Processing {len(batches)} batches with concurrency {max_concurrency}")

        async def _process_batch(batch_idx: int, batch_variables: List[Dict[str, Any]]) -> tuple[int, List[Any]]:
            """Process a single batch with retry logic"""
            last_exc = None

            for attempt in range(retries):
                try:
                    async with sem:
                        # Process the entire batch in a single API call
                        result = await self.prompt_dict({'input_str': str(batch_variables)})
                        batch_results = result

                        # Validate IDs if item_id_field is specified
                        if item_id_field:
                            sent_ids = [var.get(item_id_field) for var in batch_variables]
                            received_ids = []

                            for result in batch_results:
                                if hasattr(result, item_id_field):
                                    received_ids.append(getattr(result, item_id_field))
                                elif isinstance(result, dict) and item_id_field in result:
                                    received_ids.append(result[item_id_field])
                                else:
                                    raise ValueError(f"Result missing required ID field '{item_id_field}': {result}")

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
                    self.logger.warning(f"Batch {batch_idx} attempt {attempt + 1}/{retries} failed: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

            # If all retries failed, raise the last exception
            raise last_exc or RuntimeError(f"Unknown error processing batch {batch_idx}")

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
            else: # return unflattened results
                return batch_results
        else: # return unflattened results
            return batch_results


    async def filter_dataframe_chunk(self,
                               input_df: pd.DataFrame,
                               input_vars: Optional[Dict[str, Any]] = None,
                               item_list_field: str = 'results_list',
                               item_id_field: str = 'id',
                               retries: int = 3
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
                # Use existing prompt_dict method with retry logic
                result = await self.prompt_dict(input_dict)
                # Validate item count and IDs if item_list_field is specified
                if item_list_field:
                    if hasattr(result, item_list_field):
                        result_list = getattr(result, item_list_field)
                        if isinstance(result_list, list):
                            received_count = len(result_list)
                            if received_count != expected_count:
                                error_msg = f"Item count mismatch: expected {expected_count}, got {received_count}"
                                self.logger.warning(f"Attempt {attempt + 1}/{retries}: {error_msg}")
                                if attempt < retries - 1:
                                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                    continue
                                else:
                                    raise ValueError(error_msg)

                            # Validate IDs if item_id_field is specified and exists in DataFrame
                            if item_id_field and item_id_field in input_df.columns:
                                sent_ids = input_df[item_id_field].tolist()
                                received_ids = []

                                for item in result_list:
                                    if hasattr(item, item_id_field):
                                        received_ids.append(getattr(item, item_id_field))
                                    elif isinstance(item, dict) and item_id_field in item:
                                        received_ids.append(item[item_id_field])
                                    else:
                                        error_msg = f"Result item missing required ID field '{item_id_field}': {item}"
                                        self.logger.warning(f"Attempt {attempt + 1}/{retries}: {error_msg}")
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
                                        error_msg = f"ID presence mismatch:"
                                        if missing_ids:
                                            error_msg += f" Missing IDs: {missing_ids}"
                                        if extra_ids:
                                            error_msg += f" Extra IDs: {extra_ids}"
                                    else:
                                        error_msg = f"ID order mismatch: sent {sent_ids} != received {received_ids}"

                                    self.logger.warning(f"Attempt {attempt + 1}/{retries}: {error_msg}")
                                    if attempt < retries - 1:
                                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                        continue
                                    else:
                                        raise ValueError(error_msg)

                        else:
                            raise ValueError(f"Field '{item_list_field}' is not a list: {type(result_list)}")
                    else:
                        raise ValueError(f"Result missing required field '{item_list_field}': {result}")

                return result

            except asyncio.TimeoutError as e:
                last_exc = e
                self.logger.error(f"Timeout error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except (ConnectionError, TimeoutError) as e:
                last_exc = e
                self.logger.error(f"Network/timeout error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except ValueError as e:
                last_exc = e
                self.logger.error(f"Invalid data in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                last_exc = e
                self.logger.error(f"Unexpected error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        # If we get here, all retries failed
        raise last_exc or RuntimeError(f"Unknown error after {retries} attempts")

    async def _process_indexed_chunk(self,
                                   chunk_idx: int,
                                   chunk_df: pd.DataFrame,
                                   input_vars: Optional[Dict[str, Any]] = None,
                                   item_list_field: str = 'results_list',
                                   item_id_field: str = 'id',
                                   retries: int = 3) -> tuple[int, Any]:
        """
        Process a single chunk and return with its index for order preservation.

        Args:
            chunk_idx: Index of this chunk in the original chunk list
            chunk_df: DataFrame chunk to process
            input_vars: Optional additional variables for prompt substitution
            item_list_field: Name of the field in response containing results list
            item_id_field: Name of the ID field to validate matches
            retries: Number of retry attempts for validation failures

        Returns:
            Tuple of (chunk_index, result) for order preservation
        """
        result = await self.filter_dataframe_chunk(
            chunk_df,
            input_vars=input_vars,
            item_list_field=item_list_field,
            item_id_field=item_id_field,
            retries=retries
        )
        return chunk_idx, result

    async def filter_dataframe_batch(self,
                                   input_df: pd.DataFrame,
                                   input_vars: Optional[Dict[str, Any]] = None,
                                   item_list_field: str = 'results_list',
                                   item_id_field: str = 'id',
                                   retries: int = 3,
                                   chunk_size: int = 25,
                                   return_series: bool = False,
                                   value_field: str = 'output'
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

        Returns:
            If return_series=True: pandas Series with values for DataFrame column assignment
            Otherwise: Single concatenated result object (if item_list_field specified) or list of results
        """
        if input_df.empty:
            return []

        # Create chunks using the async generator
        chunks = []
        async for chunk in paginate_df_async(input_df, chunk_size):
            chunks.append(chunk)

        if not chunks:
            return []

        # Process all chunks concurrently with index tracking
        tasks = [
            self._process_indexed_chunk(
                i, chunk,
                input_vars=input_vars,
                item_list_field=item_list_field,
                item_id_field=item_id_field,
                retries=retries
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
                            self.logger.error(f"Field '{item_list_field}' is not a list: {type(result_list)}")
                            return [result for _, result in sorted_results]  # Fall back to returning raw results
                    else:
                        self.logger.error(f"Result missing field '{item_list_field}': {chunk_result}")
                        return [result for _, result in sorted_results]  # Fall back to returning raw results

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
                return [result for _, result in sorted_results]  # Fall back to returning raw results
        else:
            # No item_list_field specified, return list of results
            if return_series:
                self.logger.warning("return_series=True but no item_list_field specified, returning raw results")
            return [result for _, result in sorted_results]

    async def filter_dataframe(self,
                                    input_df: pd.DataFrame,
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
                     (item_list_field, item_id_field, retries, chunk_size, etc.)

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

            # With ID validation
            df["sentiment"] = await agent.filter_dataframe(
                df[['text', 'id']],
                item_id_field='id',
                value_field='sentiment'
            )
        """
        # Call filter_dataframe_batch with all provided arguments
        result = await self.filter_dataframe_batch(input_df, **kwargs)

        # Extract values from the structured result
        if hasattr(result, 'results_list'):
            # Standard case: structured object with results_list
            values = [getattr(item, value_field) for item in result.results_list]
        elif isinstance(result, list):
            # Fallback case: result is already a list of items
            values = [getattr(item, value_field) for item in result]
        else:
            # Unexpected result format
            raise ValueError(f"Unexpected result format from filter_dataframe_batch: {type(result)}")

        # Validate that we have the right number of values
        if len(values) != len(input_df):
            raise ValueError(f"Value count mismatch: expected {len(input_df)}, got {len(values)}")

        return pd.Series(values, index=input_df.index)
