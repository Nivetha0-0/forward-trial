import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import streamlit as st
from typing import Optional

# ✅ Load Firebase config from Streamlit secrets
def get_firebase_config():
    if "FIREBASE_SERVICE_ACCOUNT_KEY" in st.secrets:
        return dict(st.secrets["FIREBASE_SERVICE_ACCOUNT_KEY"])
    
    raise ValueError("❌ FIREBASE_SERVICE_ACCOUNT_KEY not found in Streamlit secrets.")

# ✅ Initialize Firebase WITHOUT projectId override
def initialize_firebase():
    if firebase_admin._apps:
        print("ℹ️ Firebase already initialized, using existing app")
        return firestore.client()

    try:
        firebase_config = get_firebase_config()
        cred = credentials.Certificate(firebase_config)

        firebase_admin.initialize_app(cred)  # ✅ Do NOT override projectId

        db = firestore.client()

        # ✅ Show project ID in logs and sidebar
        project_id = firebase_admin.get_app().project_id
        print(f"✅ Firebase initialized successfully")
        print(f"🔍 Connected to Firebase project: {project_id}")
        st.sidebar.success(f"🔗 Firebase: {project_id}")

        return db

    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        raise

# ✅ Initialize Firestore
db = initialize_firebase()

# ✅ Save unanswered question to Firestore
def save_unanswered_question(question_english):
    try:
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        qn_list = data.get("qn", [])

        if question_english not in qn_list:
            qn_list.append(question_english)
            doctor_doc_ref.set({"qn": qn_list}, merge=True)
            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️ Question already exists in unanswered list: {question_english}")

    except Exception as e:
        print(f"❌ Error saving unanswered question: {e}")
        raise

# ✅ Save user interaction to Firestore
def save_user_interaction(question_english, answer_english, user_session_id=None):
    try:
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }

        # Flag if the question is being forwarded to doctor
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
        print(f"❌ Error saving user interaction: {e}")
        raise
