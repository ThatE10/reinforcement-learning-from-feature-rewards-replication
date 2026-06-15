import json
import re
from typing import TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences from a string."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


def validate_gemini_response(text: str, model_class: Type[T]) -> T:
    """Strip markdown fences, parse JSON, and validate against a Pydantic model.

    Args:
        text:        Raw .text from a Gemini GenerateContentResponse.
        model_class: Pydantic model to validate against.  If the model defines
                     a model_validator that accepts a list, pass a raw-array
                     response and it will be wrapped automatically.

    Returns:
        A validated instance of model_class.

    Raises:
        ValueError:              If the text is not valid JSON after fence removal.
        pydantic.ValidationError: If the parsed JSON doesn't match the schema.
    """
    try:
        data = json.loads(strip_json_fences(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini response is not valid JSON: {e}") from e
    return model_class.model_validate(data)
