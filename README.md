# GuardianAI

GuardianAI is an AI-powered content moderation system for the GDG Solution Challenge. It detects harmful user-generated content (UGC) like hate speech, misinformation, and cyberbullying in real time using Python-based Google Cloud Functions, Pub/Sub, Firestore, and a React frontend.

**Live Demo**: [GuardianAI Demo](https://guardianai-455109.web.app)  
**Video**: [Watch the 3-Minute Demo](https://youtube.com/your-video-link)  
**Date**: April 06, 2025

## Overview

- **Ingestion**: UGC ingested via Pub/Sub topic `guardianai-posts`.
- **Preprocessing**: Python Cloud Function with NLTK (text) and Vision API (memes).
- **Detection**: DistilBERT on Vertex AI for text, combined with Vision API for images.
- **Actioning**: Auto-remove (>0.9 confidence), human review (0.6–0.9), or pass (<0.6).
- **Human Review**: React frontend with Gemini-generated summaries.
- **Continuous Learning**: Feedback stored in GCS, retrains model via Vertex AI.

## Repository Structure
GuardianAI/
├── cloud-functions/        # Python Cloud Functions
│   ├── preprocess_content/    # Preprocessing logic
│   ├── detect_harmful_content/ # Detection logic
│   ├── apply_actioning_logic/  # Actioning logic
│   ├── generate_review_summary/ # Summary logic
│   └── collect_training_feedback/ # Feedback logic
├── frontend/              # React frontend
├── pubsub/                # Pub/Sub utilities
├── bigquery/              # BigQuery schema and queries
├── docs/                  # Flowchart and docs
├── scripts/               # Deployment and testing
├── .env.example          # Env vars template
└── README.md             # This file

# Prerequisites
Python 3.8+ (for cloud functions)
Node.js 16+ (for frontend)
Google Cloud SDK
A Google Cloud project with Pub/Sub and BigQuery enabled
Environment variables configured (see .env.example)

## Setup Instructions
# Clone the Repository

# GuardianAI
Features
Content Preprocessing: Cleans and prepares content for analysis.
Harmful Content Detection: Identifies potentially harmful content using AI models.
Actioning Logic: Applies predefined actions based on detection results.
Review Summaries: Generates concise summaries for human review.
Feedback Collection: Gathers training feedback to improve model performance.
Scalable Architecture: Utilizes Google Cloud Functions, Pub/Sub, and BigQuery.

