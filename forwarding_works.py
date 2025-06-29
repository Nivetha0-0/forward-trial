import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore # This imports the firestore module

# --- Imports specifically for Pylance type hinting (only for Client, which is usually stable) ---
# For the Firestore Client type, this import is generally stable for Pylance.
from google.cloud.firestore_v1.client import Client as FirestoreClientType
# --- End Pylance type hinting imports ---

# Removed: from google.cloud.firestore_v1.field_value import FieldValue as FirestoreFieldValueType
# This import path seems to be causing issues with Pylance's resolution.

import datetime
from typing import Optional, Any

# Global variable to hold the Firestore client
# Use the explicitly imported type for Pylance. Pylance should now recognize 'db'.
db: Optional[FirestoreClientType] = None

def initialize_firebase_app(service_account_key_path: str):
    """
    Initializes the Firebase Admin SDK.
    Expects the path to the service account JSON file.
    """
    global db
    if not firebase_admin._apps: # Check if app is already initialized
        try:
            cred = credentials.Certificate(service_account_key_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print(f"Firebase Admin SDK initialized successfully from path: {service_account_key_path}")
        except Exception as e:
            print(f"Error initializing Firebase Admin SDK in forwarding_works.py: {e}")
            raise
    else:
        if db is None:
            db = firestore.client()
        print("Firebase Admin SDK already initialized.")


def save_unanswered_question(question_text: str, timestamp: datetime.datetime):
    """
    Saves an unanswered question to Firestore.
    Requires Firebase to be initialized.
    """
    if db is None:
        print("Firestore client not initialized. Cannot save unanswered question.")
        return

    try:
        doc_ref = db.collection('unanswered_questions').document()
        doc_ref.set({
            'question': question_text,
            'timestamp': timestamp
        })
        print(f"Unanswered question saved: {question_text}")
    except Exception as e:
        print(f"Error saving unanswered question to Firestore: {e}")

def save_user_interaction(user_input_english: str, bot_response_english: str, session_id: Optional[str] = None):
    """
    Saves a user interaction (question and bot response) to Firestore.
    Requires Firebase to be initialized.
    """
    if db is None:
        print("Firestore client not initialized. Cannot save user interaction.")
        return

    try:
        if session_id is None:
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        doc_ref = db.collection('user').document()
        doc_ref.set({
            'session_id': session_id,
            'user_input': user_input_english,
            'bot_response': bot_response_english,
            # This line works at runtime, but Pylance might complain.
            # Adding '# type: ignore' tells Pylance to suppress the warning for this line.
            'timestamp': firestore.FieldValue.server_timestamp() # type: ignore
        })
        print(f"User interaction saved: Session {session_id}, User: {user_input_english}")
    except Exception as e:
        print(f"Error saving user interaction to Firestore: {e}")

def get_firestore_client():
    """Returns the initialized Firestore client."""
    return db
