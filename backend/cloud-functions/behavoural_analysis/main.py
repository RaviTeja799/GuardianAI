import os
from datetime import datetime, timedelta, timezone, date
from collections import defaultdict
import json

import functions_framework
from google.cloud import firestore, bigquery, scheduler_v1
from google.cloud.bigquery import SchemaField
import google.cloud.scheduler_v1 as scheduler_v1_module

# --- Configuration ---
PROJECT_ID = "guardianai-455109"
FIRESTORE_COLLECTION_USER_STATS = "user_stats"
FIRESTORE_COLLECTION_BEHAVIORAL_FLAGS = "behavioral_flags"
BIGQUERY_DATASET = "guardian_ai_dataset"
BIGQUERY_TABLE = "user_behavior_analysis"
FLAG_THRESHOLD_COUNT = 5
FLAG_THRESHOLD_HOURS = 24
SCHEDULER_LOCATION = "us-central1"

# --- Initialize Clients ---
try:
    db = firestore.Client(project=PROJECT_ID)
    bq_client = bigquery.Client(project=PROJECT_ID)
    scheduler_client = scheduler_v1.CloudSchedulerClient()
    print(f"Cloud Scheduler version: {scheduler_v1_module.__version__}")
except Exception as e:
    print(f"Error initializing clients: {e}")
    db = None
    bq_client = None
    scheduler_client = None

# --- Helper Functions ---
def analyze_user_behavior(user_id: str):
    """Analyzes user behavior and returns relevant metrics."""
    try:
        # Construct and execute BigQuery query to retrieve user's posts
        posts_ref = bq_client.query(f"""
            SELECT
                DATE(TIMESTAMP_TRUNC(TIMESTAMP(timestamp_str), DAY)) as post_date,
                label,
                EXTRACT(HOUR FROM TIMESTAMP(timestamp_str)) as post_hour,
                content
            FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.posts`
            WHERE user_id = '{user_id}'
        """).result()

        # Initialize data structures for analysis
        daily_post_counts = defaultdict(int)
        label_counts = defaultdict(int)
        hourly_counts = defaultdict(int)
        keyword_counts = defaultdict(int)

        # Define harmful keywords
        harmful_keywords = ["offensive", "hate", "bully"]

        # Iterate through query results to populate data structures
        for row in posts_ref:
            daily_post_counts[row.post_date] += 1
            label_counts[row.label] += 1
            hourly_counts[row.post_hour] += 1
            if row.content:
                content_lower = row.content.lower()
                for keyword in harmful_keywords:
                    if keyword in content_lower:
                        keyword_counts[keyword] += 1

        return daily_post_counts, label_counts, hourly_counts, keyword_counts

    except Exception as e:
        print(f"Error analyzing user behavior: {e}")
        return {}, {}, {}, {}

def flag_user_if_needed(user_id: str, label_counts: dict, keyword_counts: dict):
    """Flags a user if they exceed thresholds."""
    # Define harmful labels
    harmful_labels = {"misinformation", "cyberbullying", "hate_speech", "offensive_jokes"}
    harmful_count = sum(label_counts.get(label, 0) for label in harmful_labels)
    harmful_keywords_count = sum(keyword_counts.values())

    try:
        # Get a reference to the user's stats document in Firestore
        user_stats_ref = db.collection(FIRESTORE_COLLECTION_USER_STATS).document(user_id)
        user_stats = user_stats_ref.get().to_dict()

        # Check if the user has been flagged recently
        if user_stats and user_stats.get("last_flagged_post_timestamp"):
            threshold_time = datetime.now(timezone.utc) - timedelta(hours=FLAG_THRESHOLD_HOURS)
            if user_stats["last_flagged_post_timestamp"] < threshold_time:
                print(f"User {user_id} within threshold. Skipping flag.")
                return

        # Check if the user has exceeded the threshold for harmful content
        if harmful_count >= FLAG_THRESHOLD_COUNT or harmful_keywords_count >= FLAG_THRESHOLD_COUNT:
            # Flag the user in Firestore
            flag_ref = db.collection(FIRESTORE_COLLECTION_BEHAVIORAL_FLAGS).document(user_id)
            reason = f"Exceeded threshold. Harmful posts: {harmful_count}, Harmful keywords: {harmful_keywords_count}"
            flag_data = {
                "is_flagged": True,
                "reason": reason,
                "last_checked_timestamp": firestore.SERVER_TIMESTAMP,
            }
            flag_ref.set(flag_data, merge=True)
            print(f"Flagged user {user_id}: {reason}")
            return
        print(f"User {user_id} did not exceed threshold. No flag set.")

    except Exception as e:
        print(f"Error flagging user: {e}")

def store_behavior_analysis(user_id: str, daily_posts: dict, label_dist: dict, hourly_dist: dict, keyword_counts: dict):
    """Stores behavior analysis results in BigQuery, creating the table if it doesn't exist."""
    try:
        # Get a reference to the BigQuery table
        table_ref = bq_client.dataset(BIGQUERY_DATASET).table(BIGQUERY_TABLE)
        # Define the schema for the table
        schema = [
            SchemaField("user_id", "STRING", mode="REQUIRED"),
            SchemaField("analysis_timestamp", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("daily_post_counts", "STRING"),
            SchemaField("label_distribution", "STRING"),
            SchemaField("hourly_distribution", "STRING"),
            SchemaField("keyword_counts", "STRING")
        ]

        try:
            # Attempt to get the table
            table = bq_client.get_table(table_ref)
            print(f"Table {table_ref.path} already exists.")
        except Exception:
            # Table doesn't exist, create it
            table = bigquery.Table(table_ref, schema=schema)
            table = bq_client.create_table(table)
            print(f"Created table {table_ref.path}")

        # Convert datetime.date keys to strings
        daily_posts_str_keys = {str(key): value for key, value in daily_posts.items()}

        # Create a row to insert into BigQuery
        row = {
            "user_id": user_id,
            "analysis_timestamp": datetime.now(timezone.utc),
            "daily_post_counts": json.dumps(daily_posts_str_keys, default=str),
            "label_distribution": json.dumps(label_dist),
            "hourly_distribution": json.dumps(hourly_dist),
            "keyword_counts": json.dumps(keyword_counts)
        }
        # Insert the row into BigQuery
        bq_client.insert_rows(table, [row])
        print(f"Stored behavior analysis for user {user_id} in BigQuery.")

    except Exception as e:
        print(f"Error storing behavior analysis: {e}")

# --- Cloud Function Entry Point ---
@functions_framework.http
def behavioral_analysis(request):
    """Analyzes user behavior and flags users if needed."""
    # Check if clients were initialized successfully
    if not db or not bq_client:
        return "Clients failed to initialize. Exiting.", 500

    try:
        # Query Firestore for all users in the user_stats collection
        users_query = db.collection(FIRESTORE_COLLECTION_USER_STATS).stream()
        # Iterate through each user
        for user_doc in users_query:
            user_id = user_doc.id
            # Analyze user behavior
            daily_posts, label_dist, hourly_dist, keyword_counts = analyze_user_behavior(user_id)
            # Flag user if needed based on analysis
            flag_user_if_needed(user_id, label_dist, keyword_counts)
            # Store behavior analysis results in BigQuery
            store_behavior_analysis(user_id, daily_posts, label_dist, hourly_dist, keyword_counts)

        return "Behavior analysis completed successfully.", 200

    except Exception as e:
        print(f"Error during main analysis: {e}")
        return f"Error: {e}", 500

# --- Cloud Scheduler Setup ---
def setup_scheduler():
    """Sets up a Cloud Scheduler job to trigger the Cloud Function every 6 hours."""
    # Check if scheduler client was initialized successfully
    if not scheduler_client:
        print("Scheduler client failed to initialize. Skipping scheduler setup.")
        return

    # Define the parent for the scheduler job
    parent = f"projects/{PROJECT_ID}/locations/{SCHEDULER_LOCATION}"
    # Create a new scheduler job
    job = scheduler_v1.Job()
    job.name = f"{parent}/jobs/behavior-analysis-trigger"
    job.schedule = "0 */6 * * *"
    job.time_zone = "UTC"
    # Configure the job's HTTP target
    job.http_target = scheduler_v1.HttpTarget()
    job.http_target.uri = f"https://{SCHEDULER_LOCATION}-{PROJECT_ID}.cloudfunctions.net/behavioral_analysis"
    job.http_target.http_method = scheduler_v1.HttpMethod.POST

    try:
        # Create the scheduler job
        response = scheduler_client.create_job(parent=parent, job=job)
        print(f"Created scheduler job: {response.name}")
    except Exception as e:
        print(f"Error creating scheduler job: {e}")

setup_scheduler()
