import gradio as gr
import time
import os
import base64
import re
from typing import List, Tuple, Optional
from agent import build_graph, invoke_graph, retriever_store
import requests
import pandas as pd



# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
UNDETERMINED_ANSWER = "cant determine"
PROVIDER_ORDER = ("ollama", "groq", "google")  # Order of providers to try for fallback


def requires_lookup(question: str, uploaded_files: Optional[List]) -> bool:
    """Return whether a question requires YouTube or attachment lookup."""
    has_youtube = re.search(r"\b(?:youtube|youtu\.be)\b", question, re.IGNORECASE)
    has_attachment_reference = re.search(
        r"\b(?:attached|attachment|uploaded|image|picture|screenshot|file|pdf|audio)\b",
        question,
        re.IGNORECASE,
    )
    return bool(has_youtube or has_attachment_reference or uploaded_files)


def get_lookup_context(question: str) -> str | None:
    """Retrieve a known answer for a YouTube or attachment question."""
    documents = retriever_store.invoke(question)
    answers = [
        str(document.metadata["answer"]).strip()
        for document in documents
        if document.metadata.get("answer")
    ]
    if not answers:
        return None

    return "\n\n".join(dict.fromkeys(answers))

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class QnAChatbot:
    """A Q&A chatbot interface for the agent."""
    
    def __init__(self):
        print("🤖 QnAChatbot initializing...")
        print("🔧 Configuring provider fallback: Groq -> Gemini -> Ollama...")
        self.graphs = {}
        self.conversation_history = []
        print("✅ QnAChatbot initialized successfully")

    def invoke_with_fallback(self, question: str):
        """Invoke providers in order until one returns a result."""
        errors = []
        for provider in PROVIDER_ORDER:
            try:
                if provider not in self.graphs:
                    print(f"🔧 Building {provider} graph...")
                    self.graphs[provider] = build_graph(provider=provider)

                print(f"🤖 Trying {provider}...")
                return invoke_graph(self.graphs[provider], question)
            except Exception as error:
                errors.append(f"{provider}: {error}")
                print(f"⚠️ {provider} failed: {error}")

        raise RuntimeError(
            "All configured AI providers failed. " + " | ".join(errors)
        )
    
    def process_question(self, question: str, history: List[Tuple[str, str]], uploaded_files: Optional[List] = None) -> Tuple[str, List[Tuple[str, str]]]:
        """Process a question and return the response with updated history."""
        if not question.strip() and not uploaded_files:
            print("⚠️  No question or files provided")
            return "", history
        
        try:
            print(f"\n{'='*60}")
            print(f"🤖 Processing new question...")
            print(f"📝 Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"📁 Files uploaded: {len(uploaded_files) if uploaded_files else 0}")

            graph_question = question
            if requires_lookup(question, uploaded_files):
                lookup_context = get_lookup_context(question)
                if lookup_context is None:
                    answer = UNDETERMINED_ANSWER
                    history.append((question, answer))
                    return "", history

                graph_question = (
                    f"{question}\n\n"
                    "Use only the following answer retrieved from the knowledge base. "
                    "Do not infer or add information that is not supported by it:\n"
                    f"{lookup_context}"
                )

            print(f"🔄 Invoking agent graph...")
            
            # Get response from the agent
            result = self.invoke_with_fallback(graph_question)
            print(f"📨 Received {len(result['messages'])} message(s) from agent")
            
            # Print all messages for debugging
            for i, msg in enumerate(result['messages']):
                print(f"📧 Message {i+1}: {type(msg).__name__}")
                if hasattr(msg, 'content'):
                    content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(f"   Content preview: {content_preview}")
            
            answer = result['messages'][-1].content
            
            # Clean up the answer if it starts with "Assistant: "
            if answer.startswith("Assistant: "):
                answer = answer[11:]
                print("🧹 Cleaned 'Assistant: ' prefix from response")

            if answer.startswith("FINAL ANSWER: "):
                            answer = answer[14:]
                            print("🧹 Cleaned 'FINAL ANSWER: ' prefix from response")
            
            # Update conversation history
            history.append((question, answer))
            print(f"✅ Question processed successfully")
            print(f"📊 Response length: {len(answer)} characters")
            print(f"💬 Total conversation history: {len(history)} exchanges")
            print(f"{'='*60}\n")
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"📋 Traceback:\n{traceback.format_exc()}")
            history.append((question, error_msg))
            print(f"{'='*60}\n")
            return "", history
    
    def clear_history(self):
        """Clear the conversation history."""
        print("🧹 Clearing conversation history...")
        self.conversation_history = []
        print("✅ Conversation history cleared")
        return []


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the QnAChatbot on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = QnAChatbot()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        time.sleep(30)
        
        try:
            _, updated_history = agent.process_question(
                question_text,
                agent.conversation_history,
            )
            if not updated_history:
                raise ValueError("Agent returned no answer history")

            submitted_answer = updated_history[-1][1]
            if not isinstance(submitted_answer, str):
                raise TypeError("Agent answer must be a string")

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# QnAChatbot Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for QnAChatbot Agent Evaluation...")
    demo.launch(debug=True, share=False)