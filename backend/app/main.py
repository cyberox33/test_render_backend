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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessingStatus:
    STARTED = "STARTED"
    PIPELINE_RUNNING = "PIPELINE_RUNNING"
    GENERATING_RECOMMENDATIONS = "GENERATING_RECOMMENDATIONS"
    READY = "READY"
    ERROR = "ERROR"
    NOT_FOUND = "NOT_FOUND" # For cases where session doesn't exist

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

def update_session_status(session_id: str, new_status: str):
    """Updates the processing_status for a given session_id."""
    try:
        print(f"Updating session {session_id} status to: {new_status}")
        supabase.table("user_sessions")\
            .update({"processing_status": new_status})\
            .eq("session_id", session_id)\
            .execute()
        print(f"Session {session_id} status update successful.")
    except Exception as e:
        print(f"ERROR updating session {session_id} status to {new_status}: {e}")
        # Decide if you want to raise this error or just log it
        # Raising might stop the background task prematurely if not handled

async def run_full_pipeline_and_generate_report(session_id: str):
    """
    Wrapper function for the background task.
    Runs potentially blocking pipeline and recommendation functions in threads.
    """
    print(f"BACKGROUND TASK STARTED for session {session_id}")
    current_status = ProcessingStatus.PIPELINE_RUNNING # Should already be set by caller

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
             current_status = ProcessingStatus.ERROR
             update_session_status(session_id, current_status)
             # Optionally update session status in DB to indicate failure
             return # Stop processing

        print(f"Iterative RAG pipeline finished. Updating status and starting recommendation generation...")
        current_status = ProcessingStatus.GENERATING_RECOMMENDATIONS
        update_session_status(session_id, current_status)
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
            current_status = ProcessingStatus.READY
            update_session_status(session_id, current_status)
            # Optionally update session status in DB to "ready"
        else:
            print(f"Recommendation generation/upload failed for session {session_id}.")
            current_status = ProcessingStatus.ERROR
            update_session_status(session_id, current_status)
            # Optionally update session status in DB to indicate failure

    except Exception as e:
        print(f"ERROR in background task for session {session_id}: {e}")
        traceback.print_exc()
        current_status = ProcessingStatus.ERROR
        update_session_status(session_id, current_status)
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
# Public Root Endpoint
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
    new_session_data = {
        "user_id": user_id,
        "processing_status": ProcessingStatus.STARTED # Initial status
    }
   
    try:
        print(f"Creating new session for user_id {user_id} with data: {new_session_data}")
        result = supabase.table("user_sessions").insert(new_session_data).execute()

        # Check response structure for errors more robustly if needed
        if not result.data:
             print(f"ERROR: Supabase insert for session creation returned no data. Result: {result}")
             raise HTTPException(status_code=500, detail="Failed to create session (no data returned).")

        session_id = result.data[0].get("session_id")
        if not session_id:
            print(f"ERROR: Supabase insert result missing session_id. Result: {result}")
            raise HTTPException(status_code=500, detail="Failed to retrieve session ID after creation.")

        print(f"Session created successfully: {session_id}")
        return {"session_id": session_id}

    except Exception as e:
         print(f"!!! EXCEPTION during session creation for user {current_user.username}: {e}")
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Internal server error during session creation: {str(e)}")

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
    session_id = response_data.session_id
    print(f"--- Received POST /survey-responses for session: {session_id} ---")
    try:
        update_session_status(session_id, ProcessingStatus.PIPELINE_RUNNING)
        # Store initial survey responses (keep existing logic)
        #insert_response = supabase.table("survey_responses").insert(response_data.dict()).execute()
        payload_to_insert = response_data.dict() # Contains session_id and responses list
        insert_response = supabase.table("survey_responses").insert(payload_to_insert).execute()
        print(f"Supabase survey_responses insert response object: {insert_response}")
        if not insert_response.data:
             raise HTTPException(status_code=500, detail="Failed to store survey response.")

        # --- Trigger the WRAPPER background task ---
        print(f"Survey submitted for {session_id}. Starting full background processing.")
        background_tasks.add_task(run_full_pipeline_and_generate_report, session_id)
        # ---

        return {"message": "Survey response submitted. Assessment process started in background.", "data": insert_response.data} # Return quickly
    except Exception as e:
        print(f"Error in /survey-responses for session {response_data.session_id}: {e}")
        traceback.print_exc()
        update_session_status(session_id, ProcessingStatus.ERROR)
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
    Checks status in DB and optionally file storage to report overall progress.
    """
    print(f"Checking recommendation status for session: {session_id}")
    try:
        # 1. Check session status in the database
        session_res = supabase.table("user_sessions")\
            .select("processing_status")\
            .eq("session_id", session_id)\
            .maybe_single()\
            .execute()

        if not session_res.data:
            print(f"Session not found in DB for status check: {session_id}")
            # Return "not_found" status using the constant
            return RecommendationStatus(status=ProcessingStatus.NOT_FOUND)

        db_status = session_res.data.get("processing_status")
        print(f"DB status for session {session_id}: {db_status}")

        # 2. Determine response based on DB status
        if db_status == ProcessingStatus.READY:
            # If DB says ready, check storage and generate URL
            bucket_name = "recommendations"
            remote_path = f"{session_id}/recommendation_report.pptx"
            print(f"DB status is READY. Checking storage at {bucket_name}/{remote_path}...")

            # Check existence (optional but good)
            list_response = supabase.storage.from_(bucket_name).list(path=session_id, options={"limit": 1, "search": "recommendation_report.pptx"})
            file_exists = any(file['name'] == 'recommendation_report.pptx' for file in list_response)

            if not file_exists:
                 print(f"ERROR: DB status is READY but report file not found in storage for session {session_id}")
                 # Update DB status back to ERROR? Or just report error here?
                 # update_session_status(session_id, ProcessingStatus.ERROR) # Optional correction step
                 return RecommendationStatus(status=ProcessingStatus.ERROR, error_message="Report marked ready but file is missing.")

            # File exists, generate Signed URL
            print(f"Report file found. Generating signed URL...")
            signed_url_res = supabase.storage.from_(bucket_name).create_signed_url(remote_path, 3600) # Expires in 1 hour

            if "error" in signed_url_res and signed_url_res["error"]:
                error_detail = signed_url_res['error'].get('message', 'Unknown error')
                print(f"Error generating signed URL for {remote_path}: {error_detail}")
                return RecommendationStatus(status=ProcessingStatus.ERROR, error_message=f"Failed to generate download URL: {error_detail}")

            download_url = signed_url_res.get("signedURL")
            if not download_url:
                print(f"Signed URL result missing 'signedURL' key for {remote_path}")
                return RecommendationStatus(status=ProcessingStatus.ERROR, error_message="Download URL could not be created.")

            # Return ready status with URL
            return RecommendationStatus(status=ProcessingStatus.READY, url=download_url)

        elif db_status in [ProcessingStatus.STARTED, ProcessingStatus.PIPELINE_RUNNING, ProcessingStatus.GENERATING_RECOMMENDATIONS]:
            # These states all map to "generating" for the frontend
            return RecommendationStatus(status="generating")

        elif db_status == ProcessingStatus.ERROR:
            return RecommendationStatus(status=ProcessingStatus.ERROR, error_message="Processing failed during generation.")

        else:
            # Unknown or unexpected status
            print(f"WARNING: Unknown DB processing_status '{db_status}' for session {session_id}")
            return RecommendationStatus(status=ProcessingStatus.ERROR, error_message="Unknown processing state.")

    except Exception as e:
        print(f"!!! EXCEPTION checking recommendation status for session {session_id}: {e}")
        traceback.print_exc()
        # Return a generic error status
        return RecommendationStatus(status=ProcessingStatus.ERROR, error_message=f"Server error checking status: {str(e)}")

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
