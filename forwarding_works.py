# forwarding.py
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import streamlit as st
import json
import tempfile
import os

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    try:
        # Check if Firebase service account key is in Streamlit secrets
        if "FIREBASE_SERVICE_ACCOUNT_KEY" in st.secrets:
            firebase_sa_key_json_content = st.secrets["FIREBASE_SERVICE_ACCOUNT_KEY"]
            
            # Create a temporary file with the JSON content
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
                # If it's stored as a string, use it directly
                if isinstance(firebase_sa_key_json_content, str):
                    temp_file.write(firebase_sa_key_json_content)
                else:
                    # If it's stored as a dict/object, convert to JSON string
                    temp_file.write(json.dumps(firebase_sa_key_json_content))
                temp_file.flush()
                
                # Initialize Firebase with the temporary file
                cred = credentials.Certificate(temp_file.name)
                firebase_admin.initialize_app(cred)
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
        
        # Fallback: Check if Google Application Credentials path is provided
        elif "GOOGLE_APPLICATION_CREDENTIALS" in st.secrets:
            firebase_config_path = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
            if os.path.exists(firebase_config_path):
                cred = credentials.Certificate(firebase_config_path)
                firebase_admin.initialize_app(cred)
            else:
                raise FileNotFoundError(f"Firebase config file not found at: {firebase_config_path}")
        
        else:
            raise ValueError("Firebase credentials not found in Streamlit secrets. Please add either 'FIREBASE_SERVICE_ACCOUNT_KEY' (JSON content) or 'GOOGLE_APPLICATION_CREDENTIALS' (file path) to your secrets.")
            
    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        # Re-raise the exception to prevent the app from continuing with invalid Firebase setup
        raise e

db = firestore.client()

def save_unanswered_question(question_english):
    """
    Save unanswered question to Firebase for doctor review
    
    Args:
        question_english (str): The user's question in English
    """
    try:
        # Reference to the doctor document
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        
        # Get current data
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        
        # Get existing questions list
        qn_list = data.get("qn", [])
        
        # Add new question if not already present
        if question_english not in qn_list:
            qn_list.append(question_english)
            
            # Update the document
            doctor_doc_ref.set({
                "qn": qn_list
            }, merge=True)
            
            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️  Question already exists in unanswered list: {question_english}")
            
    except Exception as e:
        print(f"❌ Error saving unanswered question: {str(e)}")
        raise e

def save_user_interaction(question_english, answer_english, user_session_id=None):
    """
    Save user question and bot response to Firebase
    
    Args:
        question_english (str): User's question in English
        answer_english (str): Bot's answer in English
        user_session_id (str, optional): User session identifier
    """
    try:
        # Create document data
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }
        
        # Check if this should be marked as unanswered
        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified", 
            "check back in a few days",
            "unable to answer your question"
        ]
        
        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"
        
        # Add to user collection
        db.collection("user").add(interaction_data)
        
        print(f"✅ User interaction saved: Q: {question_english[:50]}...")
        
    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        raise e
