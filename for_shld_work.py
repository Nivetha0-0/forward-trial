import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Optional 
import json
import tempfile
import os

# Global variable for Firestore client, initialized once
db = None

# Global variable to track if Firebase has been initialized
_firebase_app_initialized = False

def initialize_firebase_app(service_account_key_json: str):
    """
    Initializes the Firebase Admin SDK if it hasn't been initialized already,
    using the provided service account key JSON content.
    Sets the global Firestore client 'db'.
    """
    global db
    global _firebase_app_initialized

    if not _firebase_app_initialized:
        # Write the JSON content to a temporary file
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
                temp_file.write(service_account_key_json)
                temp_file.flush()
                temp_file_path = temp_file.name

            cred = credentials.Certificate(temp_file_path)
            
            # Initialize without databaseURL if only using Firestore
            firebase_admin.initialize_app(cred)
            db = firestore.client() # Get Firestore client after initialization
            
            _firebase_app_initialized = True
            print("✅ Firebase Admin SDK initialized successfully (Firestore only).")

        except Exception as e:
            print(f"❌ Error initializing Firebase Admin SDK: {e}")
            _firebase_app_initialized = False # Ensure flag is false if initialization fails
            # Re-raise to propagate the error back to main.py for st.error and st.stop
            raise e
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path) # Clean up temp file
    else:
        # If already initialized, ensure 'db' is set for subsequent calls
        if db is None:
            db = firestore.client()
        print("ℹ️ Firebase Admin SDK already initialized.")

def save_unanswered_question(question_english: str, timestamp: datetime):
    """
    Save unanswered question to Firebase for doctor review.
    Collection: 'unanswered_questions'

    Args:
        question_english (str): The user's question in English.
        timestamp (datetime): The timestamp when the question was asked.
    """
    if db is None:
        print("⚠️ Firestore DB not initialized. Cannot save unanswered question.")
        return

    try:
        # Using a new collection name for clarity and separation from "DOCTOR"
        doc_ref = db.collection("unanswered_questions").document()
        doc_ref.set({
            "question": question_english,
            "timestamp": timestamp,
            "status": "pending" # You can add a status field
        })
        print(f"✅ Unanswered question saved for doctor review: {question_english}")
    except Exception as e:
        print(f"❌ Error saving unanswered question: {str(e)}")
        # Optionally re-raise if this is a critical operation
        # raise e

def save_user_interaction(question_english: str, answer_english: str, timestamp: datetime, user_session_id: Optional[str] = None):
    """
    Save user question and bot response to Firebase.
    Collection: 'user_interactions'

    Args:
        question_english (str): User's question in English.
        answer_english (str): Bot's answer in English.
        timestamp (datetime): The timestamp of the interaction.
        user_session_id (str, optional): User session identifier.
    """
    if db is None:
        print("⚠️ Firestore DB not initialized. Cannot save user interaction.")
        return

    try:
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": timestamp,
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }

        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified",
            "check back in a few days",
            "unable to answer your question"
        ]

        # Check if the answer indicates the question was forwarded to the doctor
        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"
        
        # Using a new collection name for clarity and audit
        db.collection("user_interactions").add(interaction_data)
        
        print(f"✅ User interaction saved: Q: {question_english[:50]}... A: {answer_english[:50]}...")
    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        # Optionally re-raise if this is a critical operation
        # raise e