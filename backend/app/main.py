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

# Adjust sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import pipeline functions
from pipeline_scripts.iterative_rag_pipeline import run_iterative_rag_pipeline
from pipeline_scripts.recomendation import generate_and_upload_recommendations

# Initialize Supabase client
supabase_url = config.SUPABASE_URL
supabase_key = config.SUPABASE_KEY
# Ensure Supabase client is created fresh when needed, especially in background tasks
# supabase: Client = create_client(supabase_url, supabase_key) # Avoid global client for background tasks if possible

app = FastAPI(title="Assessment Platform Backend")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://test-render-frontend-zkg5.onrender.com"], # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (Keep existing models) ---
class SurveyQuestion(BaseModel):
    question_id: Optional[str]
    content: str
    options: Optional[List[Any]]
    subjective_answer: Optional[List[str]]

class SurveyResponse(BaseModel):
    session_id: str
    responses: List[dict]

class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class FollowupResponse(BaseModel):
    session_id: str
    question_id: int
    question: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    additional_fields: Optional[Dict] = {}
    answer: Optional[Dict] = {}

class RecommendationStatus(BaseModel):
    status: str # e.g., "pipeline_running", "generating_report", "ready", "error"
    url: Optional[str] = None
    error_message: Optional[str] = None


# --- UPDATED: Background Task Wrapper ---
async def run_full_pipeline_and_generate_report(session_id: str):
    """
    Wrapper function for the background task.
    Runs potentially blocking pipeline and recommendation functions in threads.
    Updates the processing_status in the user_sessions table.
    """
    print(f"BACKGROUND TASK STARTED for session {session_id}")
    # Create a new Supabase client instance for this background task
    # This avoids potential issues with shared clients across threads/processes
    try:
        local_supabase: Client = create_client(supabase_url, supabase_key)
    except Exception as client_e:
        print(f"FATAL: Could not create Supabase client in background task for {session_id}: {client_e}")
        # Optionally try to update status using the global client if desperate, but it's risky
        return # Cannot proceed without a client

    def update_status(new_status: str, error_msg: Optional[str] = None):
         """Helper to update status, catching errors."""
         update_payload = {"processing_status": new_status}
         # Optionally add an error message column to user_sessions if needed
         # if error_msg: update_payload["error_details"] = error_msg

         try:
             local_supabase.table("user_sessions").update(update_payload).eq("session_id", session_id).execute()
             print(f"Session {session_id} status updated to: {new_status}")
         except Exception as update_e:
             # Log the error, but don't let it crash the main task if possible
             print(f"ERROR updating session {session_id} status to {new_status}: {update_e}")

    try:
        # Initial status update when task starts
        update_status("RUNNING_RAG") # State: Iterative RAG pipeline is active

        # Step 1: Run the iterative pipeline in a thread
        print(f"Running iterative RAG pipeline for session {session_id} in thread...")
        pipeline_success = await asyncio.to_thread(run_iterative_rag_pipeline, session_id)

        if not pipeline_success:
            print(f"Iterative RAG pipeline failed or did not complete for session {session_id}.")
            update_status("ERROR", "Iterative RAG pipeline failed.") # State: Error occurred
            return # Stop processing

        print(f"Iterative RAG pipeline finished for session {session_id}. Starting recommendation generation in thread...")
        update_status("GENERATING_REPORT") # State: RAG done, report generation started

        # Step 2: Generate and Upload Recommendations in a thread
        template_file = "template.pptx" # Relative name
        recommendation_success = await asyncio.to_thread(generate_and_upload_recommendations, session_id, template_file)

        if recommendation_success:
            print(f"Recommendations generated and uploaded successfully for session {session_id}.")
            update_status("COMPLETE") # State: Process finished successfully
        else:
            print(f"Recommendation generation/upload failed for session {session_id}.")
            update_status("ERROR", "Recommendation generation or upload failed.") # State: Error occurred

    except Exception as e:
        print(f"ERROR in background task execution for session {session_id}: {e}")
        traceback.print_exc()
        update_status("ERROR", f"Unhandled exception: {str(e)}") # State: Error occurred
    finally:
        print(f"BACKGROUND TASK FINISHED for session {session_id}")
        # Clean up the local Supabase client connection if necessary (depends on library implementation)
        # await local_supabase.aclose() # If using async client


# --- Utility Function for JWT Token Creation (Unchanged) ---
def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# --- Registration Endpoint (Unchanged) ---
@app.post("/register", response_model=Token)
def register(user: User):
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
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

# --- Login Endpoint (/token) (Unchanged) ---
@app.post("/token", response_model=Token)
def login(user: User):
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
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

# --- Public Root Endpoint (Unchanged) ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Assessment Platform API"}

# --- Protected Endpoint: Get Current User Info (Unchanged) ---
@app.get("/users/me")
def read_current_user(current_user: TokenData = Depends(get_current_user)):
    return {"user": current_user}

# --- Create Session Endpoint (UPDATED to set initial status) ---
@app.post("/create-session")
def create_session(current_user: TokenData = Depends(get_current_user)):
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    user_result = supabase.table("users").select("id").eq("username", current_user.username).maybe_single().execute()
    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found") # Use 404
    user_id = user_result.data["id"]

    # Initial status when session is created, before survey submission
    initial_status = "SESSION_CREATED" # Or "AWAITING_SURVEY"

    new_session = {
        "user_id": user_id,
        "processing_status": initial_status # Set initial status
        # Removed old columns: survey_responses, retrieved_questions, coverage_state
    }
    result = supabase.table("user_sessions").insert(new_session).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create session.")

    session_id = result.data[0].get("session_id")
    print(f"Session created: {session_id} with initial status: {initial_status}")
    return {"session_id": session_id}


# --- Protected Survey Endpoints ---
@app.get("/survey-questions", response_model=List[SurveyQuestion])
async def get_survey_questions(current_user: TokenData = Depends(get_current_user)):
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    try:
        response = supabase.table("survey_questions").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching survey questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch survey questions: {str(e)}")

@app.post("/survey-responses")
async def submit_survey_response(
    response_data: SurveyResponse,
    current_user: TokenData = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Stores initial survey response, updates session status, and triggers the background task.
    """
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    session_id = response_data.session_id
    try:
        # 1. Store initial survey responses (keep existing logic)
        insert_response = supabase.table("survey_responses").insert(response_data.dict()).execute()
        if not insert_response.data:
              raise HTTPException(status_code=500, detail="Failed to store survey response.")

        # 2. Update session status to indicate processing has started
        update_payload = {"processing_status": "STARTED"} # Or "QUEUED"
        status_update_res = supabase.table("user_sessions").update(update_payload).eq("session_id", session_id).execute()
        # Optional: Check status_update_res for errors if needed

        print(f"Session {session_id} status updated to {update_payload['processing_status']}. Starting background task.")

        # 3. Trigger the WRAPPER background task
        background_tasks.add_task(run_full_pipeline_and_generate_report, session_id)

        return {"message": "Survey response submitted. Assessment process started in background.", "data": insert_response.data}
    except Exception as e:
        print(f"Error in /survey-responses for session {session_id}: {e}")
        # Attempt to set status to ERROR if submission fails critically
        try:
            supabase.table("user_sessions").update({"processing_status": "ERROR"}).eq("session_id", session_id).execute()
        except Exception as final_e:
            print(f"Failed to update session {session_id} status to ERROR after submission failure: {final_e}")
        raise HTTPException(status_code=500, detail=f"Error processing survey response: {str(e)}")


@app.post("/followup-responses")
async def submit_followup_responses(
    followup_data: List[FollowupResponse],
    current_user: TokenData = Depends(get_current_user)
):
    """
    Receives answers to follow-up questions and updates them in the database.
    The running background task detects these updates.
    """
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    if not followup_data:
        raise HTTPException(status_code=400, detail="No followup data provided.")

    session_id = followup_data[0].session_id # Assume all items belong to the same session
    print(f"Received followup submission for session: {session_id}")

    try:
        responses_updated = []
        for item in followup_data:
            answer_to_store = item.answer if isinstance(item.answer, dict) else {}
            # Use upsert to handle potential retries or store initial baseline answers robustly
            # Update only the 'answer' and let the trigger handle 'updated_at'
            update_result = supabase.table("followup_responses")\
                .update({"answer": answer_to_store})\
                .eq("session_id", item.session_id)\
                .eq("question_id", item.question_id)\
                .execute()

            # Check if data was actually updated (optional, depends on need)
            if update_result.data:
                 responses_updated.append(update_result.data[0]) # Assuming update returns the updated row
            else:
                 # Handle case where the followup_response row didn't exist?
                 print(f"Warning: No row found to update for session {item.session_id}, question {item.question_id}")


        print(f"Finished updating answers for session: {session_id}.")
        return {"message": "Follow-up responses updated successfully.", "data": responses_updated}

    except Exception as e:
        print(f"Error in /followup-responses for session {session_id}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Error saving follow-up responses: {str(e)}")


@app.get("/followup-questions")
async def get_followup_questions(
    current_user: TokenData = Depends(get_current_user),
    session_id: str = None # Made optional for flexibility, but require it
):
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID query parameter is required.")
    try:
        # Return only unanswered follow-up questions.
        response = supabase.table("followup_responses")\
            .select("*")\
            .eq("session_id", session_id)\
            .is_("answer", None)\
            .execute()
        data_to_return = response.data or []
        # print(f"Endpoint /followup-questions for session {session_id} returning {len(data_to_return)} rows.") # Less verbose log
        return data_to_return
    except Exception as e:
        print(f"Error fetching followup questions for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch follow-up questions: {str(e)}")


# --- UPDATED: Endpoint to check recommendation status and get URL ---
@app.get("/recommendations/status/{session_id}", response_model=RecommendationStatus)
async def get_recommendation_status(session_id: str, current_user: TokenData = Depends(get_current_user)):
    """
    Checks the session's processing_status and, if complete, checks storage
    and returns the report status/URL.
    """
    supabase: Client = create_client(supabase_url, supabase_key) # Get client
    bucket_name = "recommendations" # Match bucket name used in upload
    remote_path = f"{session_id}/recommendation_report.pptx"

    try:
        # 1. Get the session status from the database
        session_res = supabase.table("user_sessions").select("processing_status").eq("session_id", session_id).maybe_single().execute()

        if not session_res.data:
             print(f"Session not found for status check: {session_id}")
             # Return a specific status the frontend can understand
             return RecommendationStatus(status="not_found", error_message="Session ID not found.")

        current_pipeline_status = session_res.data.get("processing_status")
        print(f"Session {session_id} current processing_status: {current_pipeline_status}")

        # 2. Map DB status to Frontend status
        frontend_status = "error" # Default to error
        error_message = None
        check_storage = False
        generate_url = False

        if current_pipeline_status in ["SESSION_CREATED", "STARTED", "RUNNING_RAG", None]:
            # Pipeline is in the early/iterative phase. Report not ready.
            # Frontend should wait on the Followup page.
            frontend_status = "pipeline_running" # NEW status for frontend
        elif current_pipeline_status == "GENERATING_REPORT":
            # RAG done, now generating the report. Check storage.
            frontend_status = "generating_report" # Existing status frontend understands
            check_storage = True # Check if file exists yet
        elif current_pipeline_status == "COMPLETE":
            # Process finished. Report *should* be ready.
            frontend_status = "ready" # Existing status
            check_storage = True # Verify file exists
            generate_url = True # We need the URL
        elif current_pipeline_status == "ERROR":
            frontend_status = "error" # Existing status
            error_message = "Processing failed. Please check logs or contact support." # Generic error
            # TODO: Could potentially fetch a specific error message if stored in user_sessions
        else:
            # Unknown status in DB
            print(f"Warning: Unknown processing_status '{current_pipeline_status}' for session {session_id}.")
            frontend_status = "error"
            error_message = f"Unknown processing state: {current_pipeline_status}"

        # 3. Check storage if needed
        file_exists = False
        if check_storage:
            try:
                # Use list with search for efficiency if checking only one file
                list_response = supabase.storage.from_(bucket_name).list(
                    path=session_id,
                    options={"limit": 1, "search": 'recommendation_report.pptx'}
                )
                file_exists = any(file['name'] == 'recommendation_report.pptx' for file in list_response)
                print(f"Storage check for {remote_path}: File exists = {file_exists}")

                # Handle discrepancies: If status is COMPLETE but file missing, report error.
                if current_pipeline_status == "COMPLETE" and not file_exists:
                    print(f"Error: Session {session_id} status is COMPLETE, but report file not found in storage.")
                    frontend_status = "error"
                    error_message = "Report generation completed, but the file is missing in storage."
                    generate_url = False # Can't generate URL if file is missing
                # If status is GENERATING_REPORT and file *already* exists, it means it just finished.
                # Treat it as ready.
                elif current_pipeline_status == "GENERATING_REPORT" and file_exists:
                     print(f"Info: Session {session_id} status is GENERATING_REPORT, but file found. Treating as ready.")
                     frontend_status = "ready"
                     generate_url = True # Generate the URL

            except Exception as storage_e:
                print(f"Error checking Supabase storage for session {session_id}: {storage_e}")
                frontend_status = "error"
                error_message = "Failed to check report storage."
                generate_url = False # Cannot proceed if storage check fails

        # 4. Generate Signed URL if needed
        download_url = None
        if generate_url and file_exists: # Ensure file exists before generating URL
            print(f"Attempting to generate signed URL for {remote_path}")
            try:
                # Use sync client method correctly
                signed_url_res = supabase.storage.from_(bucket_name).create_signed_url(remote_path, 3600) # Expires in 1 hour

                # The sync client returns the dict directly
                if "error" in signed_url_res and signed_url_res["error"]:
                     # Handle Supabase specific error structure if known, otherwise generic
                     error_detail = signed_url_res.get('message', signed_url_res.get('error', 'Unknown URL generation error'))
                     print(f"Error generating signed URL for {remote_path}: {error_detail}")
                     frontend_status = "error"
                     error_message = f"Report ready, but failed to create download link: {error_detail}"
                elif "signedURL" in signed_url_res:
                    download_url = signed_url_res["signedURL"]
                    # Ensure status is 'ready' if URL was generated successfully
                    frontend_status = "ready"
                else:
                     print(f"Signed URL result missing 'signedURL' key for {remote_path}. Response: {signed_url_res}")
                     frontend_status = "error"
                     error_message = "Report ready, but download link could not be created (Invalid response)."

            except Exception as url_e:
                 print(f"Exception generating signed URL for {remote_path}: {url_e}")
                 frontend_status = "error"
                 error_message = f"Server error creating download link: {str(url_e)}"


        # 5. Return the final status
        return RecommendationStatus(status=frontend_status, url=download_url, error_message=error_message)

    except Exception as e:
        print(f"General error in /recommendations/status for session {session_id}: {e}")
        traceback.print_exc()
        # Return a generic server error status
        return RecommendationStatus(status="error", error_message=f"Server error checking status: {str(e)}")


# --- Run the Application (Unchanged) ---
if __name__ == "__main__":
    port_str = os.environ.get("PORT", "10000")
    try:
        port = int(port_str)
    except ValueError:
        print(f"Warning: Invalid PORT environment variable '{port_str}'. Falling back to port 10000.")
        port = 10000
    print(f"Starting server on host 0.0.0.0 and port {port}")
    # Use the correct format for uvicorn.run when running the script directly
    # The first argument should be the path to the app instance, e.g., 'main:app'
    # Assuming your file is named 'main.py' and the FastAPI instance is 'app'
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # Adjust "main:app" if your filename/app variable is different

