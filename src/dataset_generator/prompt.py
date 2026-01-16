SYSTEM_PROMPT = """You are an expert QA generator and domain expert in {domain}."""

USER_PROMPT = """[INPUT CONTEXT]
You are given sections with section title and section content. These sections are scraped from PDFs on the web and then OCRed then cleaned. They are not 100% clean and may contain garbage. Focus only on relevant content.

[INSTRUCTIONS]
- Generate QA pairs from the provided sections
- QA must be self-contained (answerable without external knowledge)
- Vary difficulty: easy, medium, hard
- Vary question types: what, why, how, when, compare, explain, summarize
- Only generate valid, logical QA pairs
- If content has nothing related to {domain} and no QA potential (random garbage, random lists, content lists, addresses, phone numbers, contact info, etc.) skip it - even if it is a whole section
- Do not rephrase same question multiple times

[ANSWER STYLE]
- Be concise and directly address the question
- Do NOT add explanations unless explicitly required
- Avoid examples, lists, or extra background unless asked
- Keep answers brief and factual

[INPUT SECTIONS]
{sections}
"""