import os
import signal
import sys
import json
import time
from functools import wraps
from datasets import load_dataset
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from tqdm import tqdm
from pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

load_dotenv()

# Load the dataset
dataset = load_dataset("mlabonne/orpo-dpo-mix-40k")
# remove illegal questions :)
dataset = dataset.filter(
    lambda r: r["source"] != "toxic-dpo-v0.2"
)

# Initialize the gemini flash 1.5 model, switch keys when daily limit of 1.5k requests is reached
google_api_key = os.getenv('GOOGLE_API_KEY4')
Settings.llm = Gemini(api_key=google_api_key,
                      model="models/gemini-1.5-flash-002", temperature=0.67)
llm = Settings.llm

# Checkpoint setup
checkpoint_file = "translation_checkpoint.txt"
current_index = 0

# Add shutdown event
shutdown_event = threading.Event()

# Handle graceful exit


def handle_exit(signum, frame):
    print("\nSaving progress before exiting...")
    save_checkpoint(current_index)
    shutdown_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            idx = int(f.read().strip())
        print(f"Resuming from checkpoint at index {idx}")
        return idx
    return 0


def save_checkpoint(index):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


class Message(BaseModel):
    content: str
    role: str


class Record(BaseModel):
    source: str
    chosen: list[Message]
    rejected: list[Message]
    prompt: str
    question: str


# Create a structured LLM using the Record model
translate_llm = llm.as_structured_llm(output_cls=Record)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_DELAY = 60  # Maximum delay between retries


def retry_on_error(max_retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                if shutdown_event.is_set():
                    return None
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = str(e)
                    if "429" in error_message or "Resource has been exhausted" in error_message:
                        print(f"\nAttempt {
                              attempt + 1} failed with rate limit error: {error_message}")
                        print(
                            "Saving progress and signaling shutdown due to daily request limit.")
                        save_checkpoint(current_index)
                        shutdown_event.set()
                        return None
                    else:
                        if attempt < max_retries - 1:
                            print(f"\nAttempt {
                                  attempt + 1} failed: {error_message}")
                            print(f"Retrying in {int(delay)} seconds...")
                            time.sleep(delay)
                            delay = min(delay * 2, MAX_DELAY)
                        else:
                            print(f"\nAll {max_retries} attempts failed. Last error: {
                                  error_message}")
            return None
        return wrapper
    return decorator


@retry_on_error()
def translate_record(record):
    # Prepare the data using Pydantic models
    original_record = Record(
        prompt=record['prompt'],
        source=record.get('source', ''),
        question=record.get('question', ''),
        chosen=[Message(**msg) for msg in record.get('chosen', [])],
        rejected=[Message(**msg) for msg in record.get('rejected', [])]
    )

    prompt = f"""
        Translate the following content into Turkish while maintaining the original meaning and structure.

        Guidelines:
        - Translate directly without any additional notes or comments.
        - Do not mention that this is a translation.
        - Keep code snippets, technical terms, and proper names unchanged.
        - Use natural Turkish expressions.
        - Only Translate the fields ('prompt', 'question', 'chosen', 'rejected').
        - Do not include any extra text or metadata.
        - The prompt and question fields must have the same content in the translation.

        Content:
        {json.dumps(original_record.model_dump(), ensure_ascii=False)}

        Provide the translation in JSON format matching this structure:
        {json.dumps(Record.model_json_schema(), indent=2, ensure_ascii=False)}
    """

    response = translate_llm.chat([ChatMessage(
        role="user",
        content=prompt
    )])

    # Access the structured response
    try:
        translated_record = response.raw

        translated_prompt = translated_record.prompt
        translated_chosen = translated_record.chosen
        translated_rejected = translated_record.rejected

        # Construct the translated record with preserved roles
        translated_record = Record(
            source=original_record.source,
            chosen=[Message(content=msg.content, role=original_record.chosen[i].role)
                    for i, msg in enumerate(translated_chosen)],
            rejected=[Message(content=msg.content, role=original_record.rejected[i].role)
                      for i, msg in enumerate(translated_rejected)],
            prompt=translated_prompt,
            question=translated_prompt,
        )
        return translated_record.model_dump()
    except Exception as e:
        print("Error parsing the translation response:", e)
        return None


lock = threading.Lock()


def process_record(i, record):
    if shutdown_event.is_set():
        return
    translated_record = translate_record(record)
    if translated_record:
        with lock:
            with open('translated_dataset.jsonl', 'a', encoding='utf-8') as outfile:
                json.dump(translated_record, outfile, ensure_ascii=False)
                outfile.write('\n')
            current_index = i + 1
            save_checkpoint(current_index)
            pbar.update(1)
    else:
        print(f"\nFailed to translate record at index {i}")
        if shutdown_event.is_set():
            return


# Load checkpoint
current_index = load_checkpoint()

# Process and translate the dataset using threading
THREAD_DELAY = 5  # Increase delay to reduce API load

# Modify the main processing loop to handle shutdown_event
for split in dataset.keys():
    total_records = len(dataset[split])
    with tqdm(total=total_records - current_index, initial=current_index, desc=f"Translating {split} split") as pbar:
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for i in range(current_index, total_records):
                if shutdown_event.is_set():
                    executor.shutdown(wait=False)
                    break
                record = dataset[split][i]
                futures.append(executor.submit(process_record, i, record))
                # Delay of 5 seconds between starting each thread
                time.sleep(THREAD_DELAY)

            # Wait for all futures to complete or shutdown_event to be set
            try:
                for future in as_completed(futures):
                    if shutdown_event.is_set():
                        break
            except KeyboardInterrupt:
                shutdown_event.set()
                executor.shutdown(wait=False)
