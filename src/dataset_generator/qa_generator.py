from google import genai
from schemas import QAOutput
from prompt import SYSTEM_PROMPT, USER_PROMPT


def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def generate_qa(
    client: genai.Client,
    sections: str,
    domain: str,
    model: str = "gemini-2.5-flash"
) -> QAOutput:
    """Generate QA pairs from sections using Gemini."""
    
    system = SYSTEM_PROMPT.format(domain=domain)
    user = USER_PROMPT.format(domain=domain, sections=sections)
    
    response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [{"text": system + "\n\n" + user}]}
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": QAOutput.model_json_schema(),
        },
    )
    
    return QAOutput.model_validate_json(response.text)