import base64
import json
import os
import re
import string
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
import logging # Use logging
import traceback

import functions_framework
import nltk
from google.cloud import firestore
from google.cloud import vision
# from google.cloud import storage # Not used in this version
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField, Table
from google.cloud import pubsub_v1 # <-- Add Pub/Sub client

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "guardianai-455109")
FIRESTORE_COLLECTION_PROCESSED = "preprocessed_data"
FIRESTORE_COLLECTION_USER_STATS = "user_stats"
BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID", "guardian_ai_dataset")
BIGQUERY_TABLE_ID = os.environ.get("BIGQUERY_TABLE_ID", "posts")
PUBSUB_TOPIC_DETECTION = os.environ.get("PUBSUB_TOPIC_DETECTION", "detection-requests")

ENABLE_VISION_API = True
VISION_API_FEATURES = [
    {"type_": vision.Feature.Type.TEXT_DETECTION},
    {"type_": vision.Feature.Type.SAFE_SEARCH_DETECTION},
]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- NLTK Setup ---
NLTK_DATA_DIR = '/tmp/nltk_data'
nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab') # <-- Check added
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    logging.info("NLTK data (punkt, stopwords, punkt_tab) found.") # Updated log
except LookupError:
    logging.info("Downloading required NLTK data to %s...", NLTK_DATA_DIR)
    nltk.download('punkt', quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download('stopwords', quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download('punkt_tab', quiet=True, download_dir=NLTK_DATA_DIR) # <-- Download added
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    logging.info("NLTK data downloaded and path updated.")

# --- Initialize Clients ---
# (Initialization code for db, vision_client, bigquery_client, publisher remains the same as previous version)
# --- Initialize Clients ---
# These are global to be reused across function invocations
try:
    db = firestore.Client()
    logging.info("Firestore client initialized.")
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client: {e}", exc_info=True)
    db = None

try:
    vision_client = vision.ImageAnnotatorClient()
    logging.info("Vision API client initialized.")
except Exception as e:
    logging.critical(f"Failed to initialize Vision API client: {e}", exc_info=True)
    vision_client = None

try:
    bigquery_client = bigquery.Client()
    logging.info("BigQuery client initialized.")
except Exception as e:
    logging.critical(f"Failed to initialize BigQuery client: {e}", exc_info=True)
    bigquery_client = None

# --- Initialize Pub/Sub Publisher Client ---
try:
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC_DETECTION)
    logging.info(f"Pub/Sub Publisher client initialized for topic: {topic_path}")
except Exception as e:
    logging.critical(f"Failed to initialize Pub/Sub Publisher client: {e}", exc_info=True)
    publisher = None
    topic_path = None


# --- Helper Functions ---
# (preprocess_text, analyze_image_content, update_user_stats remain the same as previous version)
def preprocess_text(text: Optional[str]) -> Optional[str]:
    """Basic text preprocessing: lowercasing, remove punctuation, tokenize, remove stopwords."""
    if not text or not isinstance(text, str):
        return None
    try:
        text_lower = text.lower()
        text_no_punct = re.sub(f"[{re.escape(string.punctuation)}]+", " ", text_lower)
        text_clean = re.sub(r'\s+', ' ', text_no_punct).strip()
        tokens = nltk.word_tokenize(text_clean) # This should work now
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
        processed = " ".join(tokens)
        return processed if processed else None
    except Exception as e:
        # Log the actual error and traceback
        logging.error(f"Error during text preprocessing for text starting with '{text[:50]}...': {e}", exc_info=True)
        return None # Return None on failure

def analyze_image_content(image_uri: str) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[str]]:
    """Analyzes image using Vision API (OCR and Safe Search) directly from GCS URI."""
    if not vision_client: return None, None, "VISION_CLIENT_NOT_INITIALIZED"
    if not ENABLE_VISION_API: return None, None, "VISION_API_DISABLED"
    if not image_uri or not isinstance(image_uri, str) or not image_uri.startswith("gs://"): return None, None, f"INVALID_GCS_URI: {image_uri}"
    logging.info(f"Analyzing image content for: {image_uri}")
    try:
        image = vision.Image(); image.source.image_uri = image_uri
        features_list = [vision.Feature(type_=f['type_']) for f in VISION_API_FEATURES]
        request = vision.AnnotateImageRequest(image=image, features=features_list)
        response = vision_client.annotate_image(request=request)
        if response.error.message: logging.error(f"Vision API error for {image_uri}: {response.error.message}"); return None, None, f"VISION_API_ERROR: {response.error.message}"
        ocr_text = response.full_text_annotation.text if response.full_text_annotation else None
        safe_search = response.safe_search_annotation
        safe_search_dict = {"adult": vision.Likelihood(safe_search.adult).name, "medical": vision.Likelihood(safe_search.medical).name, "spoof": vision.Likelihood(safe_search.spoof).name, "violence": vision.Likelihood(safe_search.violence).name, "racy": vision.Likelihood(safe_search.racy).name} if safe_search else None
        logging.info(f"Vision API analysis successful for {image_uri}.")
        return ocr_text, safe_search_dict, None
    except Exception as e: logging.error(f"Exception during Vision API call for {image_uri}: {e}", exc_info=True); return None, None, f"VISION_API_EXCEPTION: {str(e)}"

def update_user_stats(user_id: str, label: Optional[str], db_batch: firestore.WriteBatch):
    """Updates user post counts in Firestore using a batch."""
    if not db: logging.error("Firestore client not available in update_user_stats."); return
    if not user_id: logging.warning(f"Invalid user_id provided to update_user_stats: {user_id}"); return
    try:
        user_stats_ref = db.collection(FIRESTORE_COLLECTION_USER_STATS).document(user_id)
        update_data: Dict[str, Any] = {"total_posts": firestore.Increment(1), "last_post_timestamp": firestore.SERVER_TIMESTAMP}
        harmful_labels = {"misinformation", "cyberbullying", "hate_speech", "offensive_jokes", "violence", "adult_content", "racy_content", "explicit_content"}
        if label and isinstance(label, str) and label in harmful_labels:
            update_data["flagged_posts_count"] = firestore.Increment(1); update_data["last_flagged_post_timestamp"] = firestore.SERVER_TIMESTAMP
            logging.info(f"Incrementing flagged count for user {user_id} due to label '{label}'.")
        db_batch.set(user_stats_ref, update_data, merge=True); logging.info(f"Batched user stats update for user id: {user_id}")
    except Exception as e: logging.error(f"Error batching user stats update for user {user_id}: {e}", exc_info=True)


# --- Cloud Function Entry Point ---
@functions_framework.cloud_event
def preprocess_content(cloud_event):
    """
    Processes incoming Pub/Sub messages for text/image content, saves to Firestore & BQ,
    and publishes a message to trigger detection if preprocessing is successful.
    """
    start_time = datetime.now(timezone.utc)
    logging.info(f"Function preprocess_content started. Event ID: {cloud_event.id if hasattr(cloud_event, 'id') else 'UNKNOWN'}")

    if not all([db, vision_client, bigquery_client, publisher, topic_path]): # Check topic_path too
        logging.critical("One or more essential clients or topic path not initialized. Exiting.")
        return

    post_id = None; processed_data = None # Initialize
    try: # Decode Pub/Sub
        if not cloud_event.data or "message" not in cloud_event.data or "data" not in cloud_event.data["message"]:
             logging.error(f"Invalid Pub/Sub message structure: {cloud_event.data}"); return
        message_data_encoded = cloud_event.data["message"]["data"]
        message_data = base64.b64decode(message_data_encoded).decode("utf-8")
        data = json.loads(message_data)
        required_fields = ["post_id", "user_id", "content_type", "content", "timestamp"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            logging.error(f"Missing fields in Pub/Sub message: {missing_fields}. Data: {data}")
            # Try to update Firestore status if possible
            post_id_for_error = str(data.get("post_id", "UNKNOWN"))
            if post_id_for_error != "UNKNOWN" and db:
                db.collection(FIRESTORE_COLLECTION_PROCESSED).document(post_id_for_error).set({"error_message": f"Missing fields: {missing_fields}", "preprocessing_status": "FAILED_MISSING_FIELDS", "preprocessing_end_time": datetime.now(timezone.utc).isoformat()}, merge=True)
            return
        post_id = str(data["post_id"]); user_id = data["user_id"]; content_type = data["content_type"]
        content = data["content"]; timestamp_str = data["timestamp"]; label = data.get("label", "unknown")
        image_size = data.get("image_size"); message_id = cloud_event.data["message"].get("messageId", "unknown_message_id")
        logging.info(f"Processing post_id: {post_id}, content_type: {content_type}")
    except Exception as e: # Broad exception for parsing
        logging.error(f"Error decoding/parsing Pub/Sub message: {e}", exc_info=True)
        post_id_for_error = str(data.get("post_id", "UNKNOWN")) if 'data' in locals() and isinstance(data, dict) else "UNKNOWN"
        if post_id_for_error != "UNKNOWN" and db:
             db.collection(FIRESTORE_COLLECTION_PROCESSED).document(post_id_for_error).set({"error_message": f"Pub/Sub parsing failed: {str(e)}", "preprocessing_status": "FAILED_PUBSUB_PARSE", "preprocessing_end_time": datetime.now(timezone.utc).isoformat()}, merge=True)
        return

    # --- Initialize Firestore Data ---
    processed_data = {
        "message_id": message_id, "user_id": user_id, "content_type": content_type, "content": content,
        "timestamp_str": timestamp_str, "label": label, "processed_text": None, "ocr_text": None,
        "safe_search_annotations": None, "preprocessing_status": "STARTED",
        "preprocessing_start_time": start_time.isoformat(), "preprocessing_end_time": None,
        "error_message": None, "vision_api_error": None, "post_id": post_id, "offensive_content": None
    }
    processed_post_ref = db.collection(FIRESTORE_COLLECTION_PROCESSED).document(post_id)
    batch = db.batch()

    # --- Main Processing Logic ---
    try:
        if content_type == "text":
            logging.info(f"Preprocessing text content for post {post_id}")
            processed_data["processed_text"] = preprocess_text(content) # Should work now
            # Check result of preprocessing
            if processed_data["processed_text"] is None and content is not None:
                 logging.warning(f"Text preprocessing resulted in None for post {post_id}.")
                 processed_data["error_message"] = "Text preprocessing failed or yielded empty result."
                 processed_data["preprocessing_status"] = "COMPLETED_EMPTY"
            else:
                 processed_data["preprocessing_status"] = "COMPLETED"

        elif content_type == "image":
            # (Image processing logic remains the same as previous version)
            image_gcs_uri = content
            logging.info(f"Preprocessing image content for post {post_id} from {image_gcs_uri}")
            if image_size: logging.info(f"Processing image {post_id} with size {image_size}")
            ocr_text, safe_search, vision_error = analyze_image_content(image_gcs_uri)
            processed_data["ocr_text"] = ocr_text
            processed_data["safe_search_annotations"] = safe_search
            processed_data["vision_api_error"] = vision_error # Store the error if Vision API failed
            if vision_error:
                 processed_data["preprocessing_status"] = "FAILED_VISION_API"
                 processed_data["error_message"] = vision_error
                 logging.error(f"Vision API processing failed for post {post_id}: {vision_error}")
            else:
                 if ocr_text:
                      processed_data["processed_text"] = preprocess_text(ocr_text) # Preprocess OCR text
                      if processed_data["processed_text"] is None: logging.warning(f"OCR text preprocessing resulted in None for post {post_id}.")
                      offensive_words = {"nigga", "fuck"}; found_offensive = False
                      for word in offensive_words:
                           if word in ocr_text.lower(): logging.warning(f"Offensive term '{word}' detected in OCR text for post {post_id}"); found_offensive = True; break
                      processed_data["offensive_content"] = found_offensive
                 else:
                      processed_data["processed_text"] = None
                 processed_data["preprocessing_status"] = "COMPLETED"
        else:
            logging.warning(f"Unknown content_type '{content_type}' for post {post_id}")
            processed_data["error_message"] = f"Unknown content_type: {content_type}"
            processed_data["preprocessing_status"] = "FAILED_UNKNOWN_TYPE"

        # --- Commit Firestore Batch ---
        processed_data["preprocessing_end_time"] = datetime.now(timezone.utc).isoformat()
        batch.set(processed_post_ref, processed_data, merge=True)
        update_user_stats(user_id, label, batch)
        logging.info(f"Committing Firestore batch for post {post_id}...")
        batch.commit() # Commit Firestore changes
        logging.info(f"Firestore batch committed successfully for post {post_id}.")

        # --- Publish to Pub/Sub (AFTER successful Firestore commit) ---
        if publisher and topic_path and processed_data.get("preprocessing_status", "").startswith("COMPLETED"): # Publish if COMPLETED or COMPLETED_EMPTY etc.
             try:
                 message_payload = {"post_id": post_id}
                 message_json = json.dumps(message_payload)
                 message_bytes = message_json.encode("utf-8")
                 publish_future = publisher.publish(topic_path, message_bytes)
                 publish_future.result(timeout=60)
                 logging.info(f"Successfully published detection request for post_id {post_id} to {topic_path}")
             except Exception as pub_e:
                 logging.error(f"Failed to publish detection request for post_id {post_id}: {pub_e}", exc_info=True)
                 # Update Firestore - crucial to show publish failed if BQ depends on it implicitly
                 try: processed_post_ref.set({"error_message": f"Pub/Sub publish failed: {str(pub_e)}", "preprocessing_status": "FAILED_PUBSUB_PUBLISH"}, merge=True)
                 except Exception as fs_e: logging.error(f"Failed update FS after PubSub fail: {fs_e}")
        # ... (logging for skipping publish remains same) ...


        # --- BigQuery Insertion ---
        if bigquery_client: # Check if client is initialized
             try:
                 logging.info(f"Preparing BigQuery insertion for post {post_id}")
                 safe_search_bq = None
                 if isinstance(processed_data.get("safe_search_annotations"), dict):
                      safe_search_bq = json.dumps(processed_data["safe_search_annotations"])

                 # --- Create row WITHOUT vision_api_error ---
                 bigquery_row = {
                    "post_id": processed_data["post_id"], "user_id": processed_data["user_id"],
                    "content_type": processed_data["content_type"], "content": processed_data["content"],
                    "timestamp_str": processed_data["timestamp_str"], "label": processed_data["label"],
                    "processed_text": processed_data.get("processed_text"), "ocr_text": processed_data.get("ocr_text"),
                    "safe_search_annotations": safe_search_bq,
                    "preprocessing_end_time": processed_data["preprocessing_end_time"],
                    "preprocessing_start_time": processed_data["preprocessing_start_time"],
                    "error_message": processed_data.get("error_message"),
                    "offensive_content": processed_data.get("offensive_content"),
                    "message_id": processed_data["message_id"],
                    # "vision_api_error": processed_data.get("vision_api_error") # <-- REMOVED
                 }

                 # --- Define schema WITHOUT vision_api_error ---
                 schema = [
                    SchemaField("post_id", "STRING", mode="REQUIRED"), SchemaField("user_id", "STRING", mode="REQUIRED"),
                    SchemaField("content_type", "STRING", mode="REQUIRED"), SchemaField("content", "STRING", mode="REQUIRED"),
                    SchemaField("timestamp_str", "STRING", mode="REQUIRED"), SchemaField("label", "STRING", mode="REQUIRED"),
                    SchemaField("processed_text", "STRING"), SchemaField("ocr_text", "STRING"),
                    SchemaField("safe_search_annotations", "STRING"), SchemaField("preprocessing_end_time", "STRING"),
                    SchemaField("preprocessing_start_time", "STRING"), SchemaField("error_message", "STRING"),
                    SchemaField("offensive_content", "BOOLEAN"), SchemaField("message_id", "STRING"),
                    # SchemaField("vision_api_error", "STRING") # <-- REMOVED
                 ]
                 table_ref = bigquery_client.dataset(BIGQUERY_DATASET_ID).table(BIGQUERY_TABLE_ID)

                 # Create table logic remains same (it uses the schema defined above)
                 try: table = bigquery_client.get_table(table_ref)
                 except Exception:
                    logging.warning(f"BQ table {BIGQUERY_TABLE_ID} not found. Creating with current schema...")
                    try: table = Table(table_ref, schema=schema); table = bigquery_client.create_table(table); logging.info(f"Created BQ table: {table.path}")
                    except Exception as create_e: logging.error(f"Failed create BQ table {table.path}: {create_e}", exc_info=True); raise

                 errors = bigquery_client.insert_rows_json(table, [bigquery_row]) # Insert row without the missing field
                 if not errors:
                    logging.info(f"BigQuery insertion successful for post {post_id}.")
                 else:
                    logging.error(f"BigQuery insertion errors for post {post_id}: {errors}")
             except Exception as bq_e:
                 logging.error(f"Error during BigQuery processing for post {post_id}: {bq_e}", exc_info=True)
        else:
             logging.warning("BigQuery client not initialized, skipping insertion.")

    except Exception as e:
        logging.critical(f"Unhandled exception during processing for post {post_id}: {e}", exc_info=True)
        if processed_post_ref and post_id:
            try: processed_post_ref.set({"error_message": f"Unhandled processing failed: {str(e)} - {traceback.format_exc()[:500]}", "preprocessing_status": "FAILED_UNHANDLED_EXCEPTION", "preprocessing_end_time": datetime.now(timezone.utc).isoformat()}, merge=True)
            except Exception as fs_update_e: logging.error(f"Failed update FS after unhandled fail: {fs_update_e}", exc_info=True)

    finally:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        logging.info(f"Function preprocess_content finished for post {post_id}. Status: {processed_data.get('preprocessing_status', 'UNKNOWN') if processed_data else 'UNKNOWN'}. Duration: {duration:.2f} seconds.")
