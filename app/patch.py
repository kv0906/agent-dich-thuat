import logging
import os
import time
from functools import wraps
from threading import Lock
from typing import Optional, Union

import httpx
import openai
import translation_agent.utils as utils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


RPM = 60
MODEL = "deepseek-chat"
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Base delay in seconds for exponential backoff
API_TIMEOUT = 120.0  # Timeout in seconds for API calls


class TranslationAPIError(Exception):
    """Raised when API call fails after all retries."""
    def __init__(self, message: str, original_error: Exception = None, is_retryable: bool = True):
        super().__init__(message)
        self.original_error = original_error
        self.is_retryable = is_retryable


def model_load(
    api_key: Optional[str] = None,
    temperature: float = TEMPERATURE,
    rpm: int = RPM,
):
    """Load the DeepSeek model."""
    global client, RPM, MODEL, TEMPERATURE
    RPM = rpm
    TEMPERATURE = temperature

    logger.info("🔧 Initializing DeepSeek client...")
    
    api_key_to_use = api_key if api_key else os.getenv("DEEPSEEK_API_KEY")
    if not api_key_to_use:
        logger.error("❌ No API key provided")
        raise ValueError(
            "DeepSeek API key is required. Either enter it in the field or set DEEPSEEK_API_KEY environment variable."
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    # Create httpx client with explicit timeout
    http_client = httpx.Client(
        timeout=httpx.Timeout(API_TIMEOUT, connect=30.0)
    )
    
    client = openai.OpenAI(
        api_key=api_key_to_use,
        base_url=base_url,
        http_client=http_client,
    )
    
    logger.info("   ✅ Client initialized")
    logger.info(f"   📍 Base URL: {base_url}")
    logger.info(f"   🤖 Model: {MODEL}")
    logger.info(f"   🌡️  Temperature: {TEMPERATURE}")
    logger.info(f"   ⚡ Rate limit: {RPM} requests/min")
    logger.info(f"   ⏱️  Timeout: {API_TIMEOUT}s")


def rate_limit(get_max_per_minute):
    def decorator(func):
        lock = Lock()
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                max_per_minute = get_max_per_minute()
                min_interval = 60.0 / max_per_minute
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed

                if left_to_wait > 0:
                    time.sleep(left_to_wait)

                ret = func(*args, **kwargs)
                last_called[0] = time.time()
                return ret

        return wrapper

    return decorator


def _is_retryable_error(e: Exception) -> bool:
    """Check if an error is retryable."""
    if isinstance(e, openai.RateLimitError):
        return True
    if isinstance(e, openai.APITimeoutError):
        return True
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.InternalServerError):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code >= 500
    return False


def _get_retry_delay(attempt: int, error: Exception) -> float:
    """Get delay before next retry with exponential backoff."""
    base_delay = RETRY_DELAY_BASE * (2 ** attempt)
    
    if isinstance(error, openai.RateLimitError):
        return max(base_delay, 10.0)
    
    return min(base_delay, 30.0)


@rate_limit(lambda: RPM)
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    json_mode: bool = False,
    **kwargs,  # Accept extra args like source_lang (ignored - we always use DeepSeek)
) -> Union[str, dict]:
    """
    Generate a completion using the DeepSeek API with automatic retry.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
        model (str, optional): The model name (ignored, uses deepseek-chat).
        temperature (float, optional): The sampling temperature.
        json_mode (bool, optional): Whether to return the response in JSON format.
        **kwargs: Extra arguments (ignored, for compatibility).

    Returns:
        Union[str, dict]: The generated completion.
    
    Raises:
        TranslationAPIError: When API call fails after all retries.
    """
    model = MODEL
    temperature = TEMPERATURE
    
    last_error = None
    prompt_preview = prompt[:100].replace('\n', ' ') + "..." if len(prompt) > 100 else prompt.replace('\n', ' ')
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                logger.warning(f"   🔄 Retry attempt {attempt + 1}/{MAX_RETRIES}")
            
            logger.info("   🌐 Calling DeepSeek API...")
            call_start = time.time()
            
            if json_mode:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    top_p=1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    top_p=1,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                )
            
            call_duration = time.time() - call_start
            result = response.choices[0].message.content
            
            # Log token usage if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.debug(f"   📊 Tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
            logger.debug(f"   ⚡ API call completed in {call_duration:.2f}s")
            
            return result
        
        except Exception as e:
            last_error = e
            
            if not _is_retryable_error(e):
                logger.error(f"   ❌ Non-retryable API error: {type(e).__name__}: {e}")
                raise TranslationAPIError(
                    f"API error (non-retryable): {e}",
                    original_error=e,
                    is_retryable=False,
                )
            
            if attempt < MAX_RETRIES - 1:
                delay = _get_retry_delay(attempt, e)
                logger.warning(f"   ⚠️ Retryable error: {type(e).__name__}. Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                error_type = type(e).__name__
                logger.error(f"   ❌ API error after {MAX_RETRIES} retries: {error_type}: {e}")
                raise TranslationAPIError(
                    f"API error after {MAX_RETRIES} retries ({error_type}): {e}",
                    original_error=last_error,
                    is_retryable=True,
                )


utils.get_completion = get_completion

# Also patch the llm module directly since translation functions import from there
import translation_agent.llm as llm_module
import translation_agent.llm.completion as completion_module
import translation_agent.translation.workflow as workflow_module

llm_module.get_completion = get_completion
completion_module.get_completion = get_completion
workflow_module.get_completion = get_completion

one_chunk_initial_translation = utils.one_chunk_initial_translation
one_chunk_reflect_on_translation = utils.one_chunk_reflect_on_translation
one_chunk_improve_translation = utils.one_chunk_improve_translation
one_chunk_translate_text = utils.one_chunk_translate_text
num_tokens_in_string = utils.num_tokens_in_string
multichunk_initial_translation = utils.multichunk_initial_translation
multichunk_reflect_on_translation = utils.multichunk_reflect_on_translation
multichunk_improve_translation = utils.multichunk_improve_translation
multichunk_translation = utils.multichunk_translation
calculate_chunk_size = utils.calculate_chunk_size
