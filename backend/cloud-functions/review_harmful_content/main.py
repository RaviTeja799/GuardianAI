import functions_framework
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import json
import logging
import threading
from google.cloud import firestore
from google.cloud import pubsub_v1
from google.api_core import exceptions
import time

# Flask app setup
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")  # For flash messages

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
)
logger = logging.getLogger(__name__)

# Config
PROJECT_ID = os.environ.get("GCP_PROJECT", "guardianai-455109")
SUBSCRIPTION_NAME = "human-review-sub"  # Matches your detect_harmful_content Pub/Sub topic
FIRESTORE_COLLECTION = "preprocessed_data"

# Clients
db = None
subscriber = None
subscription_path = None
pending_reviews = []  # In-memory for now, consider Firestore if scaling

try:
    db = firestore.Client(project=PROJECT_ID)
    logger.debug(f"Initialized Firestore client for project: {PROJECT_ID}")
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
    logger.debug(f"Initialized Pub/Sub subscriber for: {subscription_path}")
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}", exc_info=True)

def pull_messages_async():
    """Asynchronously pull messages from Pub/Sub in a background thread."""
    logger.info("Starting async pull_messages thread")
    global pending_reviews  # Declare global at the start
    while True:
        try:
            logger.debug(f"Attempting to pull messages from {SUBSCRIPTION_NAME}")
            response = subscriber.pull(
                request={"subscription": subscription_path, "max_messages": 10},
                timeout=10.0  # Short timeout to avoid blocking
            )
            if not response.received_messages:
                logger.debug("No new messages found.")
                time.sleep(5)  # Avoid tight loop
                continue

            for msg in response.received_messages:
                data = json.loads(msg.message.data.decode("utf-8"))
                review_data = {
                    "post_id": data.get("post_id", "Unknown"),
                    "content": data.get("content", "No content"),
                    "summary": data.get("summary", "No summary"),
                    "label": data.get("label", "No label"),
                    "confidence": data.get("confidence", 0.0),
                    "message_id": msg.message.message_id,
                    "ack_id": msg.ack_id
                }
                if review_data["post_id"] not in [r["post_id"] for r in pending_reviews]:
                    pending_reviews.append(review_data)
                    logger.info(f"Pulled and added review for post_id: {review_data['post_id']}")
                else:
                    logger.debug(f"Duplicate post_id {review_data['post_id']} ignored")

            ack_ids = [msg.ack_id for msg in response.received_messages]
            if ack_ids:
                subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})
                logger.info(f"Acknowledged {len(ack_ids)} messages")
        except exceptions.DeadlineExceeded:
            logger.warning("Pub/Sub pull timed out, retrying...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in pull_messages: {e}", exc_info=True)
            time.sleep(5)
        time.sleep(1)  # Small delay to prevent overwhelming

# Start pull thread
pull_thread = threading.Thread(target=pull_messages_async, daemon=True)
pull_thread.start()
logger.info("Background pull thread started")

@app.route('/', methods=['GET', 'POST'])
def review_dashboard():
    """Handle GET for dashboard and POST for review submission."""
    global pending_reviews  # Declare global at the function start for all uses
    logger.info(f"Handling request: method={request.method}, url={request.url}")
    if not db:
        logger.error("Firestore client not initialized")
        flash("Database connection unavailable.")
        return render_template('review.html', reviews=[]), 500

    if request.method == 'GET':
        logger.debug(f"Rendering dashboard with {len(pending_reviews)} reviews")
        return render_template('review.html', reviews=pending_reviews)

    elif request.method == 'POST':
        logger.debug(f"Processing POST data: {request.form}")
        post_id = request.form.get('post_id')
        decision = request.form.get('decision')
        logger.debug(f"Extracted POST data: post_id={post_id}, decision={decision}")
        if not post_id or decision not in ['APPROVE', 'REJECT']:
            logger.error(f"Invalid POST data: post_id={post_id}, decision={decision}")
            flash("Invalid submission data.")
            return redirect(url_for('review_dashboard')), 400

        try:
            doc_ref = db.collection(FIRESTORE_COLLECTION).document(post_id)
            snapshot = doc_ref.get()
            if not snapshot.exists or snapshot.to_dict().get('review_status') != 'PENDING':
                logger.warning(f"Attempted review for non-pending or non-existent post: {post_id}")
                flash(f"Review for Post ID {post_id} already completed or does not exist.")
                return redirect(url_for('review_dashboard'))

            update_data = {
                "needs_review": False,
                "review_status": "COMPLETED",
                "review_decision": "APPROVED" if decision == "APPROVE" else "REJECTED",
                "reviewer_id": "reviewer_1",  # Placeholder
                "review_timestamp": firestore.SERVER_TIMESTAMP
            }
            doc_ref.set(update_data, merge=True)
            logger.info(f"Updated {post_id} in Firestore with decision: {decision}")

            pending_reviews = [r for r in pending_reviews if r["post_id"] != post_id]
            logger.info(f"Removed {post_id} from pending_reviews, remaining: {len(pending_reviews)}")
            flash(f"Review for Post ID {post_id} submitted as {decision}.")
        except exceptions.GoogleCloudError as e:
            logger.error(f"Firestore update failed for {post_id}: {e}", exc_info=True)
            flash(f"Error submitting review for Post ID {post_id}.")
        except Exception as e:
            logger.error(f"Unexpected error processing review for {post_id}: {e}", exc_info=True)
            flash(f"Error submitting review for Post ID {post_id}.")

        return redirect(url_for('review_dashboard'))

@functions_framework.http
def review_harmful_content(request):
    """Entry point for Cloud Function, delegates to Flask app."""
    logger.debug(f"Received request: method={request.method}, path={request.path}, url={request.url}, headers={request.headers}")
    if not db or not subscriber:
        logger.error("Clients not initialized, returning 500")
        return {"error": "Internal server error"}, 500
    with app.app_context():
        try:
            return app.wsgi_app(request.environ, lambda *args: None)
        except Exception as e:
            logger.error(f"Error in Flask WSGI handling: {e}", exc_info=True)
            return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    logger.info("Running Flask app locally for development")
    app.run(debug=True, host="0.0.0.0", port=8080)
