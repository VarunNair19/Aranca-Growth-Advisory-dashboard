import json
import time
from typing import Optional, Type, TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from src.config import get_logger, settings

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient:
    """Unified LLM wrapper for Gemini (primary & fast gate) with optional Groq fallback."""

    def __init__(self):
        self.gemini_client: Optional[genai.Client] = None
        if settings.gemini_api_key:
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key)
        else:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail unless configured.")

        self.groq_client = None
        if settings.groq_api_key:
            try:
                from groq import Groq

                self.groq_client = Groq(api_key=settings.groq_api_key)
                logger.info("Groq fallback client initialized with model=%s", settings.fallback_llm_model)
            except Exception:
                logger.exception("Failed initializing Groq fallback client")

        self.calls_made: int = 0

    def reset_call_counter(self) -> None:
        self.calls_made = 0

    def _can_make_call(self, max_calls: Optional[int] = None) -> bool:
        limit = max_calls if max_calls is not None else settings.max_llm_calls_per_run
        if self.calls_made >= limit:
            logger.warning("LLM circuit breaker reached: calls_made=%d limit=%d", self.calls_made, limit)
            return False
        return True

    def _pace_call(self) -> None:
        if settings.llm_call_delay_sec > 0 and self.calls_made > 0:
            time.sleep(settings.llm_call_delay_sec)

    def _call_gemini_structured(
        self,
        prompt: str,
        response_schema: Type[SchemaT],
        model_name: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> SchemaT:
        if not self.gemini_client:
            raise RuntimeError("Gemini client is not initialized (missing GEMINI_API_KEY).")

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        )
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            raise ValueError(f"Empty response text from Gemini model={model_name}")

        parsed_dict = json.loads(raw_text)
        return response_schema.model_validate(parsed_dict)

    def _call_groq_structured(
        self,
        prompt: str,
        response_schema: Type[SchemaT],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> SchemaT:
        if not self.groq_client:
            raise RuntimeError("Groq fallback client is not configured.")

        schema_json = json.dumps(response_schema.model_json_schema())
        sys_prompt = (
            (system_instruction + "\n\n" if system_instruction else "")
            + "Respond strictly with a single valid JSON object matching this JSON Schema:\n"
            + schema_json
        )
        completion = self.groq_client.chat.completions.create(
            model=settings.fallback_llm_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        if not raw_text:
            raise ValueError(f"Empty response text from Groq model={settings.fallback_llm_model}")

        parsed_dict = json.loads(raw_text)
        return response_schema.model_validate(parsed_dict)

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[SchemaT],
        use_fast_model: bool = False,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_calls: Optional[int] = None,
    ) -> Optional[SchemaT]:
        """Generates a schema-validated object with retries and optional Groq fallback."""
        if not self._can_make_call(max_calls=max_calls):
            return None

        primary_model = settings.fast_gate_model if use_fast_model else settings.primary_llm_model
        last_error: Optional[Exception] = None

        for attempt in range(1, settings.max_validation_retries + 2):
            self._pace_call()
            self.calls_made += 1
            try:
                result = self._call_gemini_structured(
                    prompt=prompt,
                    response_schema=response_schema,
                    model_name=primary_model,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                logger.debug(
                    "Structured output succeeded model=%s schema=%s attempt=%d call_count=%d",
                    primary_model,
                    response_schema.__name__,
                    attempt,
                    self.calls_made,
                )
                return result
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Schema/JSON validation failed model=%s schema=%s attempt=%d: %s",
                    primary_model,
                    response_schema.__name__,
                    attempt,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini call failed model=%s schema=%s attempt=%d: %s",
                    primary_model,
                    response_schema.__name__,
                    attempt,
                    exc,
                )

        if self.groq_client:
            try:
                self._pace_call()
                self.calls_made += 1
                result = self._call_groq_structured(
                    prompt=prompt,
                    response_schema=response_schema,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                logger.info(
                    "Fallback Groq succeeded model=%s schema=%s call_count=%d",
                    settings.fallback_llm_model,
                    response_schema.__name__,
                    self.calls_made,
                )
                return result
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "Groq fallback failed for schema=%s: %s",
                    response_schema.__name__,
                    exc,
                )

        logger.error(
            "All structured generation attempts failed for schema=%s: %s",
            response_schema.__name__,
            last_error,
        )
        return None

    def generate_text(
        self,
        prompt: str,
        use_fast_model: bool = False,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_calls: Optional[int] = None,
    ) -> Optional[str]:
        """Generates grounded text with Gemini retries and automatic Groq fallback."""
        if not self._can_make_call(max_calls=max_calls):
            return None

        primary_model = settings.fast_gate_model if use_fast_model else settings.primary_llm_model
        last_error: Optional[Exception] = None

        if self.gemini_client:
            for attempt in range(1, settings.max_validation_retries + 2):
                self._pace_call()
                self.calls_made += 1
                try:
                    config = genai_types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=system_instruction,
                    )
                    response = self.gemini_client.models.generate_content(
                        model=primary_model,
                        contents=prompt,
                        config=config,
                    )
                    raw_text = (response.text or "").strip()
                    if not raw_text:
                        raise ValueError(f"Empty text response from Gemini model={primary_model}")
                    return raw_text
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini text call failed model=%s attempt=%d: %s",
                        primary_model,
                        attempt,
                        exc,
                    )

        if self.groq_client:
            try:
                self._pace_call()
                self.calls_made += 1
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                completion = self.groq_client.chat.completions.create(
                    model=settings.fallback_llm_model,
                    temperature=temperature,
                    messages=messages,
                )
                raw_text = (completion.choices[0].message.content or "").strip()
                if not raw_text:
                    raise ValueError(f"Empty text response from Groq model={settings.fallback_llm_model}")
                logger.info(
                    "Fallback Groq text succeeded model=%s call_count=%d",
                    settings.fallback_llm_model,
                    self.calls_made,
                )
                return raw_text
            except Exception as exc:
                last_error = exc
                logger.exception("Groq text fallback failed: %s", exc)

        logger.error("All text generation attempts failed: %s", last_error)
        return None