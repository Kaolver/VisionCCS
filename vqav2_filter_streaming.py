from datasets import load_dataset
from groq import Groq
from tqdm import tqdm
import json
import time
import os
import gc

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_answer(item):
    """
    Extract answer from various possible formats in VQAv2 dataset
    """
    # Try different answer field names and formats
    if 'answer' in item:
        answer = item['answer']
        if isinstance(answer, str):
            return answer.lower().strip()
        elif isinstance(answer, list) and len(answer) > 0:
            if isinstance(answer[0], dict) and 'answer' in answer[0]:
                return answer[0]['answer'].lower().strip()
            elif isinstance(answer[0], str):
                return answer[0].lower().strip()

    if 'answers' in item:
        answers = item['answers']
        if isinstance(answers, dict):
            if 'answer' in answers and len(answers['answer']) > 0:
                return str(answers['answer'][0]).lower().strip()
        elif isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict) and 'answer' in answers[0]:
                return answers[0]['answer'].lower().strip()
            elif isinstance(answers[0], str):
                return answers[0].lower().strip()

    if 'multiple_choice_answer' in item:
        return str(item['multiple_choice_answer']).lower().strip()

    return None

def categorize_question_with_groq(question_text, model="llama-3.1-8b-instant"):
    """
    Use Groq API to categorize questions
    """

    prompt = f"""Categorize the following visual question into ONE of these categories:
1. object_detection - questions about what objects are present in the image
2. spatial_recognition - questions about spatial relationships, locations, or positions
3. attribute_recognition - questions about attributes like color, size, material, or properties
4. other - questions that don't fit the above categories

Question: {question_text}

Respond with ONLY the category name (object_detection, spatial_recognition, attribute_recognition, or other)."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=50,
        )

        category = chat_completion.choices[0].message.content.strip().lower()

        valid_categories = ['object_detection', 'spatial_recognition', 'attribute_recognition', 'other']
        if category not in valid_categories:
            for valid_cat in valid_categories:
                if valid_cat in category:
                    return valid_cat
            return 'other'

        return category

    except Exception as e:
        print(f"Error categorizing question: {e}")
        return 'other'

def filter_and_categorize_vqav2_streaming(split='validation', output_path='vqav2_filtered.json', 
                                          batch_size=50, delay=0.05, max_samples=None):
    """
    MEMORY-EFFICIENT streaming version for M3 Mac
    Processes dataset in chunks without loading everything into memory
    """

    print(f"Loading VQAv2 dataset with STREAMING (split: {split})...")
    print("This prevents memory issues on M3 Mac\n")

    # Use streaming=True to avoid loading entire dataset into memory
    dataset = load_dataset("lmms-lab/VQAv2", split=split, streaming=True)

    # Initialize categorized data
    categorized_data = {
        'object_detection': [],
        'spatial_recognition': [],
        'attribute_recognition': [],
        'other': []
    }

    yes_no_count = 0
    total_processed = 0

    print("Filtering and categorizing yes/no questions...")
    print(f"Checkpoint will be saved every {batch_size} questions\n")

    # Process items one at a time (streaming)
    for idx, item in enumerate(dataset):
        total_processed += 1

        # Extract answer
        answer = extract_answer(item)

        # Filter for yes/no answers
        if answer and answer in ['yes', 'no']:
            yes_no_count += 1

            question = item.get('question', '')

            # Categorize the question
            category = categorize_question_with_groq(question)

            # Create entry (without image to save memory)
            json_item = {
                'index': idx,
                'question_id': item.get('question_id', idx),
                'question': question,
                'answer': answer,
                'category': category
            }

            categorized_data[category].append(json_item)

            # Progress update
            if yes_no_count % 10 == 0:
                print(f"Processed {total_processed} items, found {yes_no_count} yes/no questions")

            # Add delay to avoid rate limits
            time.sleep(delay)

            # Save checkpoint every batch_size yes/no questions
            if yes_no_count % batch_size == 0:
                checkpoint_path = f"{output_path.replace('.json', '')}_checkpoint_{yes_no_count}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(categorized_data, f, indent=2)
                print(f"✓ Checkpoint saved: {checkpoint_path}")

                # Force garbage collection to free memory
                gc.collect()

            # Stop if we've reached max_samples
            if max_samples and yes_no_count >= max_samples:
                print(f"\nReached max_samples limit of {max_samples}")
                break

        # Show progress every 1000 items
        if total_processed % 1000 == 0:
            print(f"Scanned {total_processed} items, found {yes_no_count} yes/no questions so far...")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total items scanned: {total_processed}")
    print(f"Yes/No questions found: {yes_no_count}")

    if yes_no_count == 0:
        print("\n⚠️  WARNING: No yes/no questions found!")
        print("Please run debug_vqav2_structure.py to inspect the dataset")
        return {}, []

    # Save final results
    print(f"\nSaving final results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(categorized_data, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*60}")
    for category, items in categorized_data.items():
        print(f"{category:25s}: {len(items):5d} questions")

    # Save each category separately
    print(f"\nSaving individual category files...")
    for category in ['object_detection', 'spatial_recognition', 'attribute_recognition']:
        if categorized_data[category]:
            output_file = f"vqav2_{category}.json"
            with open(output_file, 'w') as f:
                json.dump(categorized_data[category], f, indent=2)
            print(f"✓ Saved {len(categorized_data[category])} questions to {output_file}")

    return categorized_data

if __name__ == "__main__":
    # Set your Groq API key
    # Option 1: Set in terminal: export GROQ_API_KEY="your_key_here"
    # Option 2: Uncomment and set here:
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key"

    print("="*60)
    print("VQAv2 FILTER - MEMORY-EFFICIENT STREAMING VERSION")
    print("Optimized for MacBook Pro M3")
    print("="*60)
    print()

    # Process dataset with streaming
    categorized_data = filter_and_categorize_vqav2_streaming(
        split='validation',
        output_path='vqav2_filtered_categorized.json',
        batch_size=50,
        delay=0.05,
        max_samples=None  # Set to 100 for quick testing, None for full dataset
    )

    print("\n✓ Processing complete!")
