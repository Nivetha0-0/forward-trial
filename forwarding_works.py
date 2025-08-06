import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json
from typing import Optional
import streamlit as st  # ✅ Needed to access Streamlit secrets

# ✅ Use the Firebase service account key from Streamlit secrets (already set)
def get_firebase_config():
    if "FIREBASE_SERVICE_ACCOUNT_KEY" in st.secrets:
        return dict(st.secrets["FIREBASE_SERVICE_ACCOUNT_KEY"])

    firebase_key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
    if firebase_key:
        try:
            return json.loads(firebase_key)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing FIREBASE_SERVICE_ACCOUNT_KEY: {str(e)}")
            raise

    google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if google_creds_path and os.path.exists(google_creds_path):
        return google_creds_path

    raise ValueError(
        "Firebase configuration not found. Please set one of:\n"
        "1. FIREBASE_SERVICE_ACCOUNT_KEY (in Streamlit secrets or as env var)\n"
        "2. GOOGLE_APPLICATION_CREDENTIALS (path to service account key file)"
    )

# ✅ Initialize Firebase WITHOUT overriding projectId manually
def initialize_firebase():
    if firebase_admin._apps:
        print("ℹ️ Firebase already initialized, using existing app")
        return firestore.client()

    try:
        firebase_config = get_firebase_config()

        if isinstance(firebase_config, str):
            cred = credentials.Certificate(firebase_config)
        else:
            cred = credentials.Certificate(firebase_config)

        # ✅ Do NOT manually pass projectId – this was causing the bug
        firebase_admin.initialize_app(cred)

        # ✅ DEBUG project ID to confirm correct connection
        print(f"✅ Firebase initialized successfully")
        print(f"🔍 Connected to project: {firebase_admin.get_app().project_id}")
        return firestore.client()

    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        raise

def get_db():
    return initialize_firebase()

db = get_db()

def save_unanswered_question(question_english):
    try:
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        qn_list = data.get("qn", [])
        if question_english not in qn_list:
            qn_list.append(question_english)
            doctor_doc_ref.set({
                "qn": qn_list
            }, merge=True)

            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️ Question already exists in unanswered list: {question_english}")

    except Exception as e:
        print(f"❌ Error saving unanswered question: {str(e)}")
        raise e

def save_user_interaction(question_english, answer_english, user_session_id=None):
    try:
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }

        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified",
            "check back in a few days",
            "unable to answer your question"
        ]

        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"

        db.collection("user").add(interaction_data)

        print(f"✅ User interaction saved: Q: {question_english[:50]}...")

    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        raise e
