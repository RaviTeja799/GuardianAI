import json
import time
import os
from google.cloud import pubsub_v1
from google.api_core.exceptions import NotFound

# --- Configuration ---
project_id = "guardianai-455109"
topic_id = "guardianai-posts"
jsonl_file = "validation_dataset.jsonl" # Make sure this file exists
batch_size = 100  # Messages per batch (client-side setting)
start_line = 0    # Start from line index 0
end_line = 16029  # End at line index 16029 (inclusive)

# GCS bucket details based on previous context
image_bucket_name = "guardian-ai-bucket"
image_path_prefix = "images"
default_image_extension = ".png" # Assuming png based on previous context

required_fields = {"post_id", "user_id", "content_type", "content", "timestamp", "label"}

# --- Initialize Publisher with Batch Settings ---
# Set batch settings once during client initialization
batch_settings = pubsub_v1.types.BatchSettings(
    max_messages=batch_size,
    max_bytes=1024 * 1024,  # 1 MB max size per batch
    max_latency=0.5,        # Max 0.5 seconds delay before publishing a partial batch
)

# Use a context manager for the publisher to ensure graceful shutdown and batch flushing
publisher = pubsub_v1.PublisherClient(batch_settings=batch_settings)
topic_path = publisher.topic_path(project_id, topic_id)
print(f"Publisher initialized for topic: {topic_path} with batch size {batch_size}")

# --- Helper Functions ---
def validate_message(data, line_num):
    """Validates a single message dictionary."""
    if not isinstance(data, dict):
        print(f"L{line_num+1}: Invalid format - not a dictionary: {data}")
        return False
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        print(f"L{line_num+1}: Invalid message - Missing fields: {missing_fields} in post_id {data.get('post_id', 'N/A')}")
        return False
    return True

def construct_image_path(post_id):
    """Constructs the expected GCS path for an image."""
    return f"gs://{image_bucket_name}/{image_path_prefix}/{post_id}{default_image_extension}"

# --- Main Publishing Logic ---
def main():
    published_count = 0
    skipped_count = 0
    error_count = 0
    futures = {} # Dictionary to store futures {future: message_info}

    print(f"Processing lines {start_line + 1} to {end_line + 1} from '{jsonl_file}'...")

    try:
        with open(jsonl_file, "r", encoding='utf-8') as f:
            for current_line_number, line in enumerate(f):
                # Efficiently check if the line is within the desired range
                if start_line <= current_line_number <= end_line:
                    try:
                        # Attempt to load JSON
                        data = json.loads(line.strip())
                        post_id = data.get("post_id", "N/A") # Get post_id early for logging

                        # Validate required fields
                        if not validate_message(data, current_line_number):
                            skipped_count += 1
                            continue

                        # Correct image path ONLY if it's an image and path is missing/wrong
                        if data['content_type'] == 'image':
                            current_content = data.get('content', '')
                            expected_path_start = f"gs://{image_bucket_name}/{image_path_prefix}/"
                            if not current_content.startswith(expected_path_start):
                                new_path = construct_image_path(data['post_id'])
                                print(f"L{current_line_number+1}: Correcting image path for post_id {data['post_id']} from '{current_content}' to '{new_path}'")
                                data['content'] = new_path
                            # else: image path already seems correct

                        # Prepare message data (bytes)
                        message_data = json.dumps(data).encode("utf-8")

                        # Publish - the client handles batching automatically
                        future = publisher.publish(topic_path, message_data)
                        # Store future with info for potential error checking later (optional)
                        futures[future] = f"L{current_line_number+1} (post_id: {post_id})"
                        published_count += 1

                    except json.JSONDecodeError as e:
                        print(f"L{current_line_number+1}: Error decoding JSON: {e}, line: {line.strip()}")
                        skipped_count += 1
                    except Exception as e:
                        print(f"L{current_line_number+1}: Error processing or publishing message: {e}")
                        error_count += 1
                elif current_line_number > end_line:
                    # Stop reading the file once we are past the desired range
                    print(f"Reached line {current_line_number + 1}, stopping file reading.")
                    break

    except FileNotFoundError:
        print(f"Error: Input file '{jsonl_file}' not found.")
        return # Exit if file not found
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        # Continue to allow already published messages to potentially finish
        error_count += 1

    print("\nFile processing finished.")
    print(f"Attempted to publish: {published_count}")
    print(f"Skipped (invalid format/validation): {skipped_count}")
    print(f"Errors during processing/publishing: {error_count}")

    # The publisher client manages futures and batching in the background.
    # When the script exits, the client attempts to send remaining messages.
    # For critical scripts, you might block until futures complete, but for bulk loading,
    # letting it finish in the background is often acceptable.
    # Add a small delay to allow background sending to progress before script exits.
    print("Allowing time for final batches to send...")
    time.sleep(5) # Adjust delay as needed, or remove if not necessary

    # Optional: Check futures if strict confirmation is needed (can be slow)
    # print("Checking publish results (this might take time)...")
    # errors_in_futures = 0
    # for future, info in futures.items():
    #     try:
    #         future.result(timeout=30) # Check result with timeout
    #     except Exception as e:
    #         print(f"Error confirming publish for {info}: {e}")
    #         errors_in_futures += 1
    # if errors_in_futures > 0:
    #     print(f"WARNING: {errors_in_futures} messages may not have published successfully.")
    # else:
    #     print("All checked futures confirmed.")

    print("Publishing script finished.")


if __name__ == "__main__":
    # Verify topic exists before starting (optional but good practice)
    try:
        publisher.get_topic(request={"topic": topic_path})
        print(f"Topic {topic_path} verified.")
    except NotFound:
        print(f"ERROR: Topic {topic_path} not found. Please create it.")
        exit() # Stop if topic doesn't exist
    except Exception as e:
        print(f"Error verifying topic: {e}")
        # Decide if you want to proceed anyway

    main()