import functions_framework
import os
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
import traceback

# Third-party Libraries
import requests
import google.auth.transport.requests
import google.oauth2.id_token
import numpy as np

# Google Cloud Client Libraries
from google.cloud import firestore
from google.cloud import pubsub_v1
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

# Logging Configuration
logger = logging.getLogger(__name__)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()  # Allow LOG_LEVEL override
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Environment Variables
PROJECT_ID_STR = os.environ.get("GCP_PROJECT", "guardianai-455109")
REGION = os.environ.get("FUNCTION_REGION", "us-central1")
PROJECT_NUMBER = os.environ.get("GCP_PROJECT_NUMBER", "891154200436")
ENDPOINT_ID_NUM = os.environ.get("VERTEX_ENDPOINT_ID_NUM", "8879239191012048896")
DEPLOYED_MODEL_ID = os.environ.get("VERTEX_DEPLOYED_MODEL_ID", "1302803631170387968")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")
PUBSUB_TOPIC_HUMAN_REVIEW = os.environ.get("PUBSUB_TOPIC_HUMAN_REVIEW", "human-review-requests")

# Validate Vertex AI Config
VERTEX_CONFIG_VALID = all([PROJECT_NUMBER, REGION, ENDPOINT_ID_NUM, DEPLOYED_MODEL_ID])
logger.info(f"Vertex AI Config: Project={PROJECT_NUMBER}, Region={REGION}, Endpoint={ENDPOINT_ID_NUM}, DeployedModel={DEPLOYED_MODEL_ID}")
logger.info(f"Vertex AI Summarization Config: Project={PROJECT_ID_STR}, Region={REGION}, Model={GEMINI_MODEL_NAME}")

# Constants
FIRESTORE_COLLECTION_PROCESSED = "preprocessed_data"
SAFE_SEARCH_HIGH_CONF_THRESHOLD = {"LIKELY", "VERY_LIKELY"}
MIN_CONF_FOR_REVIEW = float(os.environ.get("MIN_CONF_FOR_REVIEW", "0.4"))
MAX_CONF_FOR_REVIEW = float(os.environ.get("MAX_CONF_FOR_REVIEW", "0.7"))
HARMFUL_LABEL_SET = set(os.environ.get("HARMFUL_LABELS", "cyberbullying,double_meaning,hate_speech,misinformation,explicit_content,violence,adult_content,racy_content").split(","))
logger.info(f"Review thresholds: [{MIN_CONF_FOR_REVIEW}, {MAX_CONF_FOR_REVIEW}], Harmful labels: {HARMFUL_LABEL_SET}")

# Global Clients
db = None
publisher_human_review = None
topic_path_human_review = None
vertex_sdk_initialized = False

def initialize_global_clients():
    global db, publisher_human_review, topic_path_human_review, vertex_sdk_initialized
    if db is None:
        try:
            db = firestore.Client(project=PROJECT_ID_STR)
            logger.info(f"Firestore client initialized for project: {db.project}")
        except Exception as e:
            logger.critical(f"CRITICAL: Error initializing Firestore client: {e}", exc_info=True)
            db = None
    if publisher_human_review is None and PROJECT_ID_STR and PUBSUB_TOPIC_HUMAN_REVIEW:
        try:
            publisher_human_review = pubsub_v1.PublisherClient()
            topic_path_human_review = publisher_human_review.topic_path(PROJECT_ID_STR, PUBSUB_TOPIC_HUMAN_REVIEW)
            logger.info(f"Pub/Sub Publisher initialized for topic: {topic_path_human_review}")
        except Exception as e:
            logger.critical(f"CRITICAL: Pub/Sub human review init failed: {e}", exc_info=True)
            publisher_human_review = None
            topic_path_human_review = None
    if not vertex_sdk_initialized and PROJECT_ID_STR and REGION:
        try:
            vertexai.init(project=PROJECT_ID_STR, location=REGION)
            vertex_sdk_initialized = True
            logger.info("Vertex AI SDK initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL: Vertex AI SDK init failed: {e}", exc_info=True)
            vertex_sdk_initialized = False

initialize_global_clients()

def call_vertex_ai_endpoint(text_to_classify: str) -> Optional[Dict[str, Any]]:
    logger.debug(f"START: call_vertex_ai_endpoint with text: '{text_to_classify[:100]}...'")
    try:
        if not VERTEX_CONFIG_VALID:
            logger.error("ERROR: Vertex AI config invalid.")
            return None
        if not text_to_classify or not isinstance(text_to_classify, str):
            logger.warning("WARN: No valid text provided.")
            return None

        predict_url = f"https://{ENDPOINT_ID_NUM}.{REGION}-{PROJECT_NUMBER}.prediction.vertexai.goog/v1/projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID_NUM}:predict"
        request_payload = {"instances": [{"content": text_to_classify}]}
        request_json_body = json.dumps(request_payload)
        logger.debug(f"Request URL: {predict_url}, Body: {request_json_body}")

        auth_req = google.auth.transport.requests.Request()
        creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        creds.refresh(auth_req)
        headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
        response = requests.post(predict_url, headers=headers, data=request_json_body, timeout=120)
        response.raise_for_status()
        logger.info(f"Response status: {response.status_code}, Raw: {response.text[:500]}...")

        parsed_response = json.loads(response.text)
        predictions = parsed_response.get("predictions", [])
        if not predictions or not isinstance(predictions, list):
            logger.error(f"ERROR: No valid predictions: {parsed_response}")
            return None

        first_pred = predictions[0]
        if isinstance(first_pred, str):
            inner_pred = json.loads(first_pred)
            if "predictions" in inner_pred:
                pred_dict = inner_pred["predictions"][0]
            else:
                pred_dict = inner_pred
        else:
            pred_dict = first_pred

        logger.debug(f"Parsed pred_dict: {pred_dict}")
        if 'scores' not in pred_dict or 'labels' not in pred_dict:
            logger.error(f"ERROR: Missing scores/labels in pred_dict: {pred_dict}")
            return None

        scores = pred_dict['scores']
        labels = pred_dict['labels']
        id2label = {0: 'correct_information', 1: 'cyberbullying', 2: 'double_meaning', 3: 'hate_speech', 
                    4: 'humorous', 5: 'misinformation', 6: 'safe', 7: 'neutral'}
        best_score_index = np.argmax(scores)
        predicted_label = id2label.get(best_score_index, "unknown")
        confidence = float(scores[best_score_index])
        result = {"label": predicted_label, "confidence": confidence}
        logger.info(f"Prediction: {result}")
        return result

    except Exception as e:
        logger.error(f"ERROR: Unexpected failure: {e}", exc_info=True)
        return None
    finally:
        logger.debug("END: call_vertex_ai_endpoint")

def interpret_safe_search(annotations: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    logger.debug(f"interpret_safe_search: {annotations}")
    if not annotations or not isinstance(annotations, dict):
        return None
    high_risk_categories = {cat: annotations[cat] for cat in ["adult", "violence", "racy"] if annotations.get(cat) in SAFE_SEARCH_HIGH_CONF_THRESHOLD}
    if high_risk_categories:
        label = "explicit_content"
        if "violence" in high_risk_categories:
            label = "violence"
        elif "adult" in high_risk_categories:
            label = "adult_content"
        elif "racy" in high_risk_categories:
            label = "racy_content"
        logger.info(f"SafeSearch result: {label}, categories: {high_risk_categories}")
        return {"label": label, "confidence": 0.95, "source": "SAFE_SEARCH"}
    logger.info("SafeSearch: No high-risk content.")
    return None

def get_text_summary(text_to_summarize: Optional[str]) -> Optional[str]:
    logger.debug(f"START: get_text_summary with text: '{text_to_summarize[:100] if text_to_summarize else 'None'}...'")
    if not vertex_sdk_initialized:
        logger.error("Vertex AI SDK not initialized for summarization.")
        return "[Vertex SDK Not Initialized]"
    if not text_to_summarize or not isinstance(text_to_summarize, str):
        logger.warning("No valid text provided for summarization.")
        return "[No Text Provided]"

    try:
        logger.info(f"Attempting summary with {GEMINI_MODEL_NAME}...")
        model = GenerativeModel(GEMINI_MODEL_NAME)
        prompt = (
            "Summarize the following text in one neutral sentence (max 15 words). "
            "Focus on the main subject, not judgment. If abusive/offensive, note briefly:\n\n"
            f"Text: {text_to_summarize}\n\nSummary:"
        )
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        logger.debug(f"Prompt: {prompt[:200]}..., Safety settings: {safety_settings}")
        response = model.generate_content(
            [Part.from_text(prompt)],
            generation_config={"max_output_tokens": 1000, "temperature": 0.4},
            safety_settings=safety_settings
        )
        logger.debug(f"Raw response: {response}")

        if not response.candidates:
            logger.warning(f"No candidates in response. Prompt feedback: {getattr(response, 'prompt_feedback', 'None')}")
            return "[No Candidates Returned]"

        candidate = response.candidates[0]
        logger.debug(f"Candidate: finish_reason={candidate.finish_reason}, text={candidate.text}")
        if candidate.finish_reason not in [FinishReason.STOP, FinishReason.MAX_TOKENS]:
            logger.warning(f"Summary failed with finish reason: {candidate.finish_reason.name}")
            return f"[Failed: {candidate.finish_reason.name}]"

        summary = candidate.text.strip()
        if not summary:
            logger.warning("Gemini returned empty summary.")
            return "[Empty Summary]"
        
        logger.info(f"Generated summary: '{summary}' (length: {len(summary)})")
        return summary

    except Exception as e:
        logger.error(f"ERROR: Summarization failed: {str(e)}", exc_info=True)
        return f"[Error: {str(e)}]"
    finally:
        logger.debug("END: get_text_summary")

@functions_framework.cloud_event
def detect_harmful_content(cloud_event):
    start_time = datetime.now(timezone.utc)
    event_id = cloud_event.id if hasattr(cloud_event, 'id') else 'UNKNOWN'
    logger.info(f"START: detect_harmful_content, Event ID: {event_id}")
    post_id = None
    try:
        message = cloud_event.data.get('message', {})
        message_data_encoded = message['data']
        message_data = json.loads(base64.b64decode(message_data_encoded).decode("utf-8"))
        post_id = str(message_data.get("post_id", ""))
        if not post_id:
            raise ValueError("No post_id in payload")
        logger.info(f"Triggered for post_id: {post_id}")
    except Exception as e:
        logger.critical(f"CRITICAL: Pub/Sub parse error: {e}", exc_info=True)
        return

    if not db or not vertex_sdk_initialized or not publisher_human_review:
        initialize_global_clients()
    if not db or not vertex_sdk_initialized or not publisher_human_review:
        logger.critical(f"CRITICAL: No clients initialized for {post_id}.")
        return

    doc_ref = db.collection(FIRESTORE_COLLECTION_PROCESSED).document(post_id)
    try:
        snapshot = doc_ref.get()
        if not snapshot.exists or not snapshot.to_dict():
            logger.error(f"ERROR: Document {post_id} not found or empty.")
            return
        data = snapshot.to_dict()
        logger.debug(f"Fetched data: {data}")
    except Exception as e:
        logger.error(f"ERROR: Firestore fetch failed for {post_id}: {e}", exc_info=True)
        return

    if data.get("detection_status") == "COMPLETED":
        logger.info(f"Skipping {post_id}: Already completed.")
        return
    if data.get("preprocessing_status") != "COMPLETED":
        logger.warning(f"Skipping {post_id}: Preprocessing not completed.")
        return

    detection_results = {
        "detection_status": "PENDING",
        "detection_start_time": start_time.isoformat(),
        "detection_end_time": None,
        "detection_label": None,
        "detection_confidence": None,
        "detection_model_source": None,
        "detection_error": None,
        "content_summary": None,
        "review_status": None  # New field for human review tracking
    }
    try:
        doc_ref.set({"detection_status": "PENDING", "detection_start_time": start_time.isoformat()}, merge=True)
        logger.info(f"Set PENDING for {post_id}")

        text_for_model = data.get("processed_text")
        safe_search_annotations = data.get("safe_search_annotations")
        content_type = data.get("content_type")
        logger.debug(f"Text: {text_for_model}, Annotations: {safe_search_annotations}, Type: {content_type}")

        text_model_pred = None
        safe_search_pred = None
        if text_for_model:
            text_model_pred = call_vertex_ai_endpoint(text_for_model)
            logger.info(f"Text model result: {text_model_pred}")
        if content_type == "image" and safe_search_annotations:
            safe_search_pred = interpret_safe_search(safe_search_annotations)

        final_label, final_confidence, final_source = "neutral", 0.1, "DEFAULT"
        if safe_search_pred:
            final_label, final_confidence, final_source = safe_search_pred["label"], safe_search_pred["confidence"], safe_search_pred["source"]
        elif text_model_pred:
            final_label, final_confidence, final_source = text_model_pred["label"], text_model_pred["confidence"], "TEXT_MODEL"

        logger.info(f"Final prediction for {post_id}: Label={final_label}, Conf={final_confidence}, Source={final_source}")

        summary = None
        trigger_review = (final_label in HARMFUL_LABEL_SET and 
                          final_confidence is not None and 
                          MIN_CONF_FOR_REVIEW <= final_confidence < MAX_CONF_FOR_REVIEW)
        logger.debug(f"Trigger review check: {trigger_review}, Label={final_label}, Conf={final_confidence}")

        if trigger_review:
            logger.info(f"{post_id}: Triggering human review for {final_label} (Conf: {final_confidence:.4f})")
            text_to_summarize = data.get("processed_text") or data.get("ocr_text") or data.get("content")
            summary = get_text_summary(text_to_summarize)
            logger.info(f"Summary result for {post_id}: '{summary}'")

            if publisher_human_review and topic_path_human_review and summary and not summary.startswith("["):
                review_data = {
                    "post_id": post_id,
                    "content": data.get("content"),
                    "summary": summary,
                    "label": final_label,
                    "confidence": final_confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                message = json.dumps(review_data).encode("utf-8")
                try:
                    future = publisher_human_review.publish(topic_path_human_review, message)
                    message_id = future.result(timeout=60)
                    logger.info(f"Published human review request for {post_id} with message ID: {message_id}")
                    detection_results["review_status"] = "PENDING"
                except Exception as pub_e:
                    logger.error(f"Failed to publish human review request: {pub_e}", exc_info=True)
                    detection_results["review_status"] = "FAILED_TO_PUBLISH"
            else:
                logger.warning(f"Human review not triggered for {post_id}: Publisher={publisher_human_review}, Summary={summary}")
                detection_results["review_status"] = "NOT_PUBLISHED"

        detection_results.update({
            "detection_label": final_label,
            "detection_confidence": final_confidence,
            "detection_model_source": final_source,
            "detection_status": "COMPLETED",
            "content_summary": summary
        })

    except Exception as e:
        logger.error(f"ERROR: Detection logic failed for {post_id}: {e}", exc_info=True)
        detection_results.update({
            "detection_status": "FAILED",
            "detection_error": f"{str(e)} - {traceback.format_exc()[:500]}",
            "content_summary": "[Error During Processing]"
        })

    detection_results["detection_end_time"] = datetime.now(timezone.utc).isoformat()
    try:
        logger.debug(f"Writing to Firestore: {detection_results}")
        doc_ref.set(detection_results, merge=True)
        logger.info(f"Successfully updated Firestore for {post_id}: {detection_results}")
    except Exception as e:
        logger.error(f"ERROR: Firestore update failed for {post_id}: {e}", exc_info=True)
    logger.info(f"END: detect_harmful_content for {post_id}")
