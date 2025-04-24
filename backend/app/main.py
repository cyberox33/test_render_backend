import asyncio
import traceback
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from supabase import create_client, Client
from utils import config
from utils.auth import (
    get_current_user,
    TokenData,
    get_password_hash,
    verify_password,
    SECRET_KEY,
    ALGORITHM
)
from jose import jwt
import uvicorn
import sys, os

# Adjust sys.path so that modules in the parent directory are found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import pipeline functions
from pipeline_scripts.iterative_rag_pipeline import run_iterative_rag_pipeline
from pipeline_scripts.recomendation import generate_and_upload_recommendations # Assuming this function exists now


# Initialize Supabase client
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

app = FastAPI(title="Assessment Platform Backend")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://test-render-frontend-zkg5.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Pydantic Models
# -------------------------------
class SurveyQuestion(BaseModel):
    question_id: Optional[str]
    content: str
    options: Optional[List[Any]]
    subjective_answer: Optional[List[str]]

class SurveyResponse(BaseModel):
    session_id: str
    responses: List[dict]  # e.g. {"question_id": "Q1", "answer": "1", ...}

class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FollowupResponse(BaseModel):
    session_id: str
    question_id: int  # using int to match questions.id
    question: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    additional_fields: Optional[Dict] = {}
    answer: Optional[Dict] = {}

class RecommendationStatus(BaseModel):
    status: str # e.g., "generating", "ready", "error", "not_found"
    url: Optional[str] = None
    error_message: Optional[str] = None

async def run_full_pipeline_and_generate_report(session_id: str):
    """
    Wrapper function for the background task.
    Runs potentially blocking pipeline and recommendation functions in threads.
    """
    print(f"BACKGROUND TASK STARTED for session {session_id}")
    try:
        # Step 1: Run the iterative pipeline in a thread
        print(f"Running iterative RAG pipeline for session {session_id} in thread...")
        # Assuming run_iterative_rag_pipeline is a standard synchronous function
        # Use asyncio.to_thread (requires Python 3.9+)
        pipeline_success = await asyncio.to_thread(run_iterative_rag_pipeline, session_id)

        # --- Alternative for Python < 3.9 using FastAPI's utility ---
        # from fastapi.concurrency import run_in_threadpool
        # pipeline_success = await run_in_threadpool(run_iterative_rag_pipeline, session_id=session_id)
        # ---

        if not pipeline_success: # Check if pipeline indicated success
             print(f"Iterative RAG pipeline failed or did not complete for session {session_id}.")
             # Optionally update session status in DB to indicate failure
             return # Stop processing

        print(f"Iterative RAG pipeline finished for session {session_id}. Starting recommendation generation in thread...")

        # Step 2: Generate and Upload Recommendations in a thread
        template_file = "template.pptx" # Relative name
        # Assuming generate_and_upload_recommendations is a standard synchronous function
        # Use asyncio.to_thread (requires Python 3.9+)
        recommendation_success = await asyncio.to_thread(generate_and_upload_recommendations, session_id, template_file)

        # --- Alternative for Python < 3.9 using FastAPI's utility ---
        # recommendation_success = await run_in_threadpool(generate_and_upload_recommendations, session_id=session_id, template_path_relative=template_file)
        # ---

        if recommendation_success:
            print(f"Recommendations generated and uploaded successfully for session {session_id}.")
            # Optionally update session status in DB to "ready"
        else:
            print(f"Recommendation generation/upload failed for session {session_id}.")
            # Optionally update session status in DB to indicate failure

    except Exception as e:
        print(f"ERROR in background task for session {session_id}: {e}")
        traceback.print_exc()
        # Optionally update session status in DB to "error"
    finally:
         print(f"BACKGROUND TASK FINISHED for session {session_id}")
# -------------------------------
# Utility Function for JWT Token Creation
# -------------------------------
def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# -------------------------------
# Registration Endpoint
# -------------------------------
@app.post("/register", response_model=Token)
def register(user: User):
    existing = supabase.table("users").select("*").eq("username", user.username).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Username already registered.")
    password_hash = get_password_hash(user.password)
    new_user = {"username": user.username, "password_hash": password_hash}
    result = supabase.table("users").insert(new_user).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to register user.")
    token_data = {"sub": user.username}
    access_token = create_access_token(token_data)
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------------------
# Login Endpoint (/token)
# -------------------------------
@app.post("/token", response_model=Token)
def login(user: User):
    result = supabase.table("users").select("*").eq("username", user.username).execute()
    users = result.data
    if not users:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user_data = users[0]
    if not verify_password(user.password, user_data.get("password_hash")):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token_data = {"sub": user.username}
    access_token = create_access_token(token_data)
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------------------
# Public Root Endpoint (we made mistakes)
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Assessment Platform API"}

# -------------------------------
# Protected Endpoint: Get Current User Info
# -------------------------------
@app.get("/users/me")
def read_current_user(current_user: TokenData = Depends(get_current_user)):
    return {"user": current_user}

# -------------------------------
# Create Session Endpoint
# -------------------------------
@app.post("/create-session")
def create_session(current_user: TokenData = Depends(get_current_user)):
    user_result = supabase.table("users").select("*").eq("username", current_user.username).execute()
    if not user_result.data:
        raise HTTPException(status_code=400, detail="User not found")
    user_id = user_result.data[0]["id"]
    new_session = {
        "user_id": user_id,
        "survey_responses": {},
        "retrieved_questions": {},
        "coverage_state": {}
    }
    result = supabase.table("user_sessions").insert(new_session).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create session.")
    session_id = result.data[0].get("session_id")
    return {"session_id": session_id}

# -------------------------------
# Protected Survey Endpoints
# -------------------------------
@app.get("/survey-questions", response_model=List[SurveyQuestion])
async def get_survey_questions(current_user: TokenData = Depends(get_current_user)):
    try:
        response = supabase.table("survey_questions").select("*").execute()
        return response.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/survey-responses")
async def submit_survey_response(
    response_data: SurveyResponse,
    current_user: TokenData = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Stores initial survey response and triggers the background task for
    the pipeline AND recommendation generation.
    """
    try:
        # Store initial survey responses (keep existing logic)
        insert_response = supabase.table("survey_responses").insert(response_data.dict()).execute()
        if not insert_response.data:
             raise HTTPException(status_code=500, detail="Failed to store survey response.")

        # --- Trigger the WRAPPER background task ---
        print(f"Survey submitted for {response_data.session_id}. Starting full background processing.")
        background_tasks.add_task(run_full_pipeline_and_generate_report, response_data.session_id)
        # ---

        return {"message": "Survey response submitted. Assessment process started in background.", "data": insert_response.data} # Return quickly
    except Exception as e:
        print(f"Error in /survey-responses for session {response_data.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/followup-responses")
async def submit_followup_responses(
    followup_data: List[FollowupResponse],
    current_user: TokenData = Depends(get_current_user)
    # No background_tasks needed here anymore
):
    """
    Receives answers to follow-up questions (baseline or iterative)
    and ONLY updates them in the database. The running background task
    will detect these updates via its wait_for_batch_answers function.
    """
    if not followup_data:
        raise HTTPException(status_code=400, detail="No followup data provided.")

    session_id = followup_data[0].session_id
    print(f"Received followup submission for session: {session_id}")

    try:
        responses_updated = []
        for item in followup_data:
            answer_to_store = item.answer if isinstance(item.answer, dict) else {}
            # Use upsert to handle potential retries or store initial baseline answers robustly
            update_result = supabase.table("followup_responses")\
                .update({"answer": answer_to_store})\
                .eq("session_id", item.session_id)\
                .eq("question_id", item.question_id)\
                .execute()
            responses_updated.append(update_result.data)

        print(f"Finished updating answers for session: {session_id}.")

        # --- NO TRIGGER LOGIC HERE ---

        return {"message": "Follow-up responses updated successfully.", "data": responses_updated}

    except Exception as e:
        print(f"Error in /followup-responses for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/followup-questions")
async def get_followup_questions(
    current_user: TokenData = Depends(get_current_user), 
    session_id: str = None
):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    try:
        # Return only unanswered follow-up questions.
        response = supabase.table("followup_responses")\
            .select("*")\
            .eq("session_id", session_id)\
            .is_("answer", None)\
            .execute()
        data_to_return = response.data or []
        print(f"Endpoint /followup-questions for session {session_id} returning {len(data_to_return)} rows.") # Add Log
        print(f"Data sample: {data_to_return[:2]}") # Optionally log first few rows
        return data_to_return
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: Endpoint to check recommendation status and get URL ---
@app.get("/recommendations/status/{session_id}", response_model=RecommendationStatus)
async def get_recommendation_status(session_id: str, current_user: TokenData = Depends(get_current_user)):
    """
    Checks if the recommendation report exists in storage and returns its status/URL.
    """
    bucket_name = "recommendations" # Match bucket name used in upload
    # Define expected path in the bucket
    remote_path = f"{session_id}/recommendation_report.pptx"

    try:
        # 1. Check if the file exists in the bucket
        # Listing files with a prefix is one way to check existence
        list_response = supabase.storage.from_(bucket_name).list(path=session_id, options={"limit": 1})

        file_exists = any(file['name'] == 'recommendation_report.pptx' for file in list_response)

        if not file_exists:
            # File not found. Could still be generating or failed.
            # TODO: Enhance this by checking a status field in user_sessions table if needed
            print(f"Report file not found in storage for session {session_id}")
            # Return "generating" or "not_found" based on session status if available
            # For now, assume "generating" if not found
            return RecommendationStatus(status="generating")

        # 2. File exists, generate Signed URL (assuming private bucket)
        print(f"Report file found for session {session_id}. Generating signed URL.")
        signed_url_res = supabase.storage.from_(bucket_name).create_signed_url(remote_path, 3600) # Expires in 1 hour

        if "error" in signed_url_res and signed_url_res["error"]:
             error_detail = signed_url_res['error'].get('message', 'Unknown error')
             print(f"Error generating signed URL for {remote_path}: {error_detail}")
             # If URL generation fails, report an error status
             return RecommendationStatus(status="error", error_message=f"Failed to generate download URL: {error_detail}")

        download_url = signed_url_res.get("signedURL")
        if not download_url:
             print(f"Signed URL result missing 'signedURL' key for {remote_path}")
             return RecommendationStatus(status="error", error_message="Download URL could not be created.")

        # 3. Return 'ready' status with the URL
        return RecommendationStatus(status="ready", url=download_url)

    except Exception as e:
        print(f"Error checking recommendation status for session {session_id}: {e}")
        # Return a generic error status
        return RecommendationStatus(status="error", error_message=str(e))


# -------------------------------
# Run the Application
# -------------------------------
if __name__ == "__main__":
    # 1. Get the PORT environment variable provided by Render.
    #    os.environ.get() retrieves the value as a string.
    #    If 'PORT' isn't set (e.g., running locally), use 10000 as a default.
    #    Render's default is 10000, so it's a sensible default choice.
    port_str = os.environ.get("PORT", "10000")

    # 2. Convert the port string to an integer.
    #    Uvicorn expects the port number as an integer.
    try:
        port = int(port_str)
    except ValueError:
        # Fallback if the PORT environment variable is somehow not a valid number
        print(f"Warning: Invalid PORT environment variable '{port_str}'. Falling back to port 10000.")
        port = 10000

    # 3. Run Uvicorn, binding to all interfaces ('0.0.0.0') and the port Render expects.
    #    'reload=False' is recommended for production on Render.
    print(f"Starting server on host 0.0.0.0 and port {port}") # Optional: Add a log message
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
