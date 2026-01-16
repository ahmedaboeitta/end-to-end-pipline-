import json
import os
import random
import time
from chunk_grouper import load_chunks, group_chunks, format_sections
from qa_generator import create_client, generate_qa


def extract_metadata(group: list[dict]) -> dict:
    return {
        "source_chunks": [chunk["id"] for chunk in group],
        "source_urls": list(set(chunk["source_url"] for chunk in group)),
        "source_titles": list(set(chunk["source_title"] for chunk in group)),
    }


def save_qa(qa_list: list[dict], filepath: str):
    with open(filepath, "w") as f:
        json.dump(qa_list, f, indent=2)


def generate_with_retry(client, sections, domain, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            return generate_qa(client, sections, domain, model)
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries")


def run_qa(
    chunks_path: str,
    output_dir: str,
    domain: str,
    api_key: str,
    max_tokens: int = 500,
    train_ratio: float = 0.8,
    model: str = "gemini-2.5-flash"
):
    """QA dataset generation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    all_qa_path = os.path.join(output_dir, "all_qa.json")

    chunks = load_chunks(chunks_path)
    groups = group_chunks(chunks, max_tokens)
    print(f"Loaded {len(chunks)} chunks into {len(groups)} groups")

    if os.path.exists(all_qa_path):
        with open(all_qa_path, "r") as f:
            all_qa = json.load(f)
        print(f"Resuming from {len(all_qa)} existing QA pairs")
    else:
        all_qa = []

    processed_file = os.path.join(output_dir, "processed_groups.json")
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed = set(json.load(f))
    else:
        processed = set()

    client = create_client(api_key)

    for i, group in enumerate(groups):
        if i in processed:
            print(f"Skipping group {i+1}/{len(groups)} (already done)")
            continue

        print(f"Processing group {i+1}/{len(groups)}...")

        sections = format_sections(group)
        metadata = extract_metadata(group)

        try:
            output = generate_with_retry(client, sections, domain, model)

            for qa in output.qa_pairs:
                all_qa.append({
                    "question": qa.question,
                    "answer": qa.answer,
                    **metadata
                })

            processed.add(i)
            save_qa(all_qa, all_qa_path)
            with open(processed_file, "w") as f:
                json.dump(list(processed), f)

            print(f"  Generated {len(output.qa_pairs)} QA pairs (total: {len(all_qa)})")

        except Exception as e:
            print(f"Error on group {i+1}: {e}")
            continue

    print(f"\nTotal: {len(all_qa)} QA pairs")

    random.shuffle(all_qa)
    split_idx = int(len(all_qa) * train_ratio)
    train_data = all_qa[:split_idx]
    eval_data = all_qa[split_idx:]

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, "eval.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Saved {len(train_data)} train, {len(eval_data)} eval")


def run_summarization(chunks_path, output_dir, domain, api_key, **kwargs):
    """Summarization dataset generation pipeline."""
    # TODO: implement later
    raise NotImplementedError("Summarization use case not implemented yet")


def run_other(chunks_path, output_dir, domain, api_key, **kwargs):
    """Other use case pipeline."""
    # TODO: implement later
    raise NotImplementedError("This use case not implemented yet")


def run(
    chunks_path: str,
    output_dir: str,
    domain: str,
    use_case: str,
    api_key: str,
    **kwargs
):
    """Main entry point - routes to appropriate pipeline."""
    
    if use_case.lower() == "qa":
        run_qa(chunks_path, output_dir, domain, api_key, **kwargs)
    elif use_case.lower() == "summarization":
        run_summarization(chunks_path, output_dir, domain, api_key, **kwargs)
    else:
        run_other(chunks_path, output_dir, domain, api_key, **kwargs)


if __name__ == "__main__":
    run(
        chunks_path="../../data/chunks/chunks.json",
        output_dir="../../data/qa_dataset",
        domain="pregnancy",
        use_case="qa",
        api_key=os.getenv("GEMINI_API_KEY"),
    )