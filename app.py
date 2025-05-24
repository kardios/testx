import streamlit as st
import fitz  # PyMuPDF
from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError
import base64
import os
import yaml  # For loading prompts
import time
from pathlib import Path  # For path handling
from st_copy_to_clipboard import st_copy_to_clipboard
from groq import Groq
from openai import OpenAI 

# --- Constants ---
PROMPTS_DIR = Path("./prompts")

# --- API Key Configuration & Client Initialization ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
APP_PASSWORD = os.environ.get("PASSWORD") # Password for app access

client_anthropic = None
if ANTHROPIC_API_KEY:
    client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

client_groq = None
if GROQ_API_KEY:
    client_groq = Groq(api_key=GROQ_API_KEY)

client_openai = None 
if OPENAI_API_KEY: 
    client_openai = OpenAI(api_key=OPENAI_API_KEY) 


# --- Load Prompts ---
def load_prompts_from_dir(prompts_dir: Path) -> dict:
    """Loads ALL valid prompt configurations from YAML files in a directory."""
    prompts_data = {}
    if not prompts_dir.is_dir():
        # This error will be shown in the main app area if prompts dir is missing
        # st.error(f"Prompts directory not found: {prompts_dir}") 
        return prompts_data

    loaded_files_count = 0
    for file_path in prompts_dir.glob("*.yaml"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'prompt_word' in data:
                    if 'instruction' in data:
                        prompt_name = data['prompt_word']
                        prompts_data[prompt_name] = data['instruction']
                        loaded_files_count += 1
                    else:
                        # st.warning(f"Skipping prompt file {file_path.name}: missing 'instruction' key.")
                        pass # Keep UI cleaner, log if necessary
                else:
                    # st.warning(f"Skipping prompt file {file_path.name}: missing 'prompt_word' key.")
                    pass
        except yaml.YAMLError as e:
            # st.error(f"Error parsing YAML file {file_path.name}: {e}")
            pass
        except Exception as e:
            # st.error(f"Error reading file {file_path.name}: {e}")
            pass
    
    # A warning if no prompts are loaded can be handled in the main UI part
    # if loaded_files_count == 0:
    #     st.warning(f"No valid prompt files found or loaded from {prompts_dir}")
    return prompts_data

def read_pdf_with_pymupdf(uploaded_file):
    """Reads text content from each page of an uploaded PDF using PyMuPDF."""
    text = ""
    try:
        start_time = time.time()
        uploaded_file.seek(0)
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
        end_time = time.time()
        st.info(f"PyMuPDF processing took {end_time - start_time:.2f} seconds.")
    except Exception as e:
        st.error(f"Error reading PDF with PyMuPDF: {e}")
        return None
    return text

def read_pdf_with_claude(uploaded_file, question):
    """Reads PDF content from an uploaded file using Claude API."""
    if not client_anthropic:
        st.error("Anthropic client not initialized. Please set ANTHROPIC_API_KEY.")
        return None, None
    try:
        start_time = time.time()
        uploaded_file.seek(0) 
        pdf_content = uploaded_file.read()
        pdf_data = base64.standard_b64encode(pdf_content).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document", 
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            }
        ]
        
        claude_model_pdf_extraction = "claude-3-7-sonnet-20250219" 
        
        token_count_response = client_anthropic.messages.count_tokens(
            model=claude_model_pdf_extraction,
            messages=messages,
        )
        token_count = token_count_response.input_tokens

        if token_count > 16384: 
            st.error(
                f"Error: PDF + question ({token_count} tokens) exceeds Claude's 16384 token limit for this operation."
            )
            return None, None

        response = client_anthropic.messages.create(
            model=claude_model_pdf_extraction,
            max_tokens=4096, 
            messages=messages,
        )
        extracted_text = ""
        if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
            extracted_text = response.content[0].text
        else:
            st.warning("Claude's response format was unexpected for PDF extraction.")
            return None, token_count 

        end_time = time.time()
        st.info(f"Claude (PDF processing using {claude_model_pdf_extraction}) took {end_time - start_time:.2f} seconds.")
        return extracted_text, token_count

    except Exception as e: 
        st.error(f"Error processing PDF with Claude: {e}")
        return None, None

def process_text_with_claude(text_content, prompt):
    """Sends plain text content and a prompt to Claude API for processing."""
    if not client_anthropic:
        st.error("Anthropic client not initialized. Please set ANTHROPIC_API_KEY.")
        return None, None
        
    working_text = None 
    outcome_text = None

    if not text_content or not prompt:
        st.warning("Step 2 (Claude): Missing text content or prompt.")
        return working_text, outcome_text

    try:
        start_time = time.time()
        messages = [{"role": "user", "content": f"{prompt}\n\nHere is the text to process:\n\n{text_content}"}]
        claude_model_text_processing = "claude-3-7-sonnet-20250219" 
        token_count = None
        try:
            token_count_response = client_anthropic.messages.count_tokens(model=claude_model_text_processing, messages=messages)
            token_count = token_count_response.input_tokens
            st.info(f"Step 2: Estimated input tokens for Claude ({claude_model_text_processing}): {token_count}")

            if token_count > 131072: 
                st.error(f"Step 2 Error (Claude): Text + prompt ({token_count} tokens) exceeds limit for {claude_model_text_processing}).")
                return working_text, outcome_text
        except Exception as token_error:
            st.warning(f"Step 2 (Claude): Could not estimate token count: {token_error}")

        response = client_anthropic.messages.create(
            model=claude_model_text_processing,
            max_tokens=16384, 
            thinking={"type": "enabled","budget_tokens": 8192}, 
            messages=messages,
        )
        
        if response.content and len(response.content) > 0:
            if hasattr(response.content[0], 'thinking') and response.content[0].thinking:
                if isinstance(response.content[0].thinking, str): 
                    working_text = response.content[0].thinking
                elif hasattr(response.content[0].thinking, 'text'): 
                     working_text = response.content[0].thinking.text

            if len(response.content) > 1 and hasattr(response.content[1], 'text'):
                outcome_text = response.content[1].text
            elif hasattr(response.content[0], 'text'): 
                outcome_text = response.content[0].text
            else:
                st.warning("Claude's response format (text processing) not as expected (outcome).")
        else:
            st.warning("Claude's response empty/unexpected (text processing).")

        st.success(f"Step 2: Secondary processing by Claude ({claude_model_text_processing}) successful!")
        end_time = time.time()
        st.info(f"Claude (text processing) took {end_time - start_time:.2f} seconds.")
    
    except APIStatusError as e: 
        st.error(f"Step 2 Claude API Error: Status {e.status_code} - {e.response.text if hasattr(e, 'response') and e.response else e.body if hasattr(e, 'body') else 'No response body'}")
    except APIConnectionError as e: 
        st.error(f"Step 2 Claude API Connection Error: {e}")
    except RateLimitError as e: 
        st.error(f"Step 2 Claude API Rate Limit Error: {e}")
    except Exception as e: 
        st.error(f"Step 2 Error processing text with Claude: {e}")

    return working_text, outcome_text


def process_text_with_groq(text_content, prompt, model_name="llama-3.3-70b-versatile"): 
    """Sends plain text content and a prompt to Groq API for processing."""
    if not client_groq:
        st.error("Groq client not initialized. Please set GROQ_API_KEY.")
        return None, None

    working_text = None 
    outcome_text = None

    if not text_content or not prompt:
        st.warning("Step 2 (Groq): Missing text content or prompt.")
        return working_text, outcome_text

    try:
        start_time = time.time()
        messages = [{"role": "user", "content": f"{prompt}\n\nHere is the text to process:\n\n{text_content}"}]
        
        st.info(f"Step 2: Sending text to Groq with model {model_name}...")
        chat_completion = client_groq.chat.completions.create(
            messages=messages,
            model=model_name, 
            max_tokens=4000, 
        )
        
        if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content:
            outcome_text = chat_completion.choices[0].message.content
            st.success(f"Step 2: Secondary processing by Groq ({model_name}) successful!")
            if chat_completion.usage:
                st.info(f"Groq token usage: Prompt: {chat_completion.usage.prompt_tokens}, Completion: {chat_completion.usage.completion_tokens}, Total: {chat_completion.usage.total_tokens}")
                if hasattr(chat_completion.usage, 'total_time') and chat_completion.usage.total_time and chat_completion.usage.total_time > 0 : 
                     st.info(f"Processing speed: {chat_completion.usage.total_time:.2f}s, {chat_completion.usage.completion_tokens / chat_completion.usage.total_time:.2f} tokens/sec")
                else:
                     st.info(f"Processing time not available or zero.")
        else:
            st.error("Groq API response was empty or not in the expected format.")

        end_time = time.time()
        st.info(f"Groq ({model_name} text processing) client-side time: {end_time - start_time:.2f} seconds.")

    except Exception as e: 
        st.error(f"Step 2 Error processing text with Groq ({model_name}): {e}")

    return working_text, outcome_text

def process_text_with_openai(text_content, prompt, model_name="gpt-4o", enable_web_search=False):
    """
    Sends plain text content and a prompt to OpenAI API for processing.
    Uses Responses API with web search if enabled and model is suitable, otherwise Chat Completions.
    """
    if not client_openai:
        st.error("OpenAI client not initialized. Please set OPENAI_API_KEY.")
        return None, None 

    outcome_text = None
    sources_text = "Sources: Not applicable or not provided by this model." 
    web_search_models = ["gpt-4o", "gpt-4.1"] 

    if not text_content or not prompt:
        st.warning("Step 2 (OpenAI): Missing text content or prompt for OpenAI processing.")
        return outcome_text, sources_text 

    full_prompt_for_openai = f"{prompt}\n\nHere is the text to process:\n\n{text_content}"
    start_time = time.time()
    
    use_responses_api = enable_web_search and model_name in web_search_models

    if use_responses_api:
        try:
            st.info(f"Step 2: Attempting OpenAI with Responses API (model: {model_name}, web search: ENABLED)...")
            if not hasattr(client_openai, 'responses'):
                raise AttributeError("OpenAI client does not have '.responses' attribute. Will use Chat Completions.")

            response = client_openai.responses.create(
                model=model_name,
                input=full_prompt_for_openai,
                tools=[{
                    "type": "web_search_preview", 
                    "search_context_size": "high" 
                }],
                tool_choice={"type": "web_search_preview"} 
            )
            
            if response.output_text:
                outcome_text = response.output_text
            
            openai_sources_list = []
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == "message" and \
                       hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == "output_text" and \
                               hasattr(content_item, 'annotations') and content_item.annotations:
                                for annotation in content_item.annotations:
                                    if hasattr(annotation, 'type') and annotation.type == "url_citation" and \
                                       hasattr(annotation, 'url') and annotation.url and \
                                       hasattr(annotation, 'title') and annotation.title:
                                        openai_sources_list.append(f"- [{annotation.title}]({annotation.url})")
            
            if openai_sources_list:
                sources_text = f"Sources (OpenAI {model_name} via Responses API with Web Search):\n" + "\n".join(list(set(openai_sources_list)))
            elif outcome_text: 
                 sources_text = f"Sources (OpenAI {model_name} via Responses API with Web Search): Web search tool was utilized. No specific citable annotations found."
            st.success(f"Step 2: OpenAI ({model_name} via Responses API with Web Search) successful!")

        except AttributeError as ae: 
            st.warning(f"OpenAI Responses API not available or suitable for '{model_name}'. Error: {ae}. Using Chat Completions API instead.")
            use_responses_api = False 
        except Exception as e: 
            st.error(f"Step 2 Error with OpenAI ({model_name} via Responses API): {e}")
            if "authentication" in str(e).lower():
                st.error("Please double-check your OPENAI_API_KEY.")
            outcome_text = f"Error with Responses API: {e}" 
            use_responses_api = False 

    if not use_responses_api: 
        try:
            search_status = "DISABLED" if not enable_web_search else "UNAVAILABLE for this model or due to previous error"
            st.info(f"Step 2: Using OpenAI Chat Completions API (model: {model_name}, web search: {search_status})...")
            chat_response = client_openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt_for_openai}],
                max_tokens=4000,
                temperature=0.5
            )
            if chat_response.choices and chat_response.choices[0].message and chat_response.choices[0].message.content:
                outcome_text = chat_response.choices[0].message.content
                sources_text = f"Sources (OpenAI {model_name} via Chat Completions): Information likely from training data."
                st.success(f"Step 2: OpenAI ({model_name} via Chat Completions) successful!")
                if chat_response.usage:
                    st.info(f"OpenAI Token Usage (Chat): Prompt: {chat_response.usage.prompt_tokens}, Completion: {chat_response.usage.completion_tokens}, Total: {chat_response.usage.total_tokens}")
            else:
                st.error(f"OpenAI API ({model_name} via Chat Completions) response was empty or not in the expected format.")

        except Exception as fallback_e:
            st.error(f"Step 2 Error with OpenAI ({model_name} via Chat Completions): {fallback_e}")
            if "authentication" in str(fallback_e).lower():
                st.error("Please double-check your OPENAI_API_KEY.")
            outcome_text = f"Error during Chat Completions: {fallback_e}"


    end_time = time.time()
    st.info(f"OpenAI ({model_name} text processing) took {end_time - start_time:.2f} seconds.")
    
    return outcome_text, sources_text

def check_password():
    """Returns `True` if the user has entered the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # If APP_PASSWORD is not set, bypass password protection
    if not APP_PASSWORD:
        st.session_state.password_correct = True
        return True

    if st.session_state.password_correct:
        return True

    # Show input for password.
    password_placeholder = st.empty()
    password_attempt = password_placeholder.text_input("Enter Password to access the App", type="password", key="password_input_field")

    if password_attempt:
        if password_attempt == APP_PASSWORD:
            st.session_state.password_correct = True
            password_placeholder.empty()  # Clear the password input
            st.rerun() # Rerun to show the app
        else:
            st.error("😕 Password incorrect. Please try again.")
            st.session_state.password_correct = False # Ensure it's false on incorrect attempt
    return False


def main_app_content():
    """Encapsulates the main content of the Streamlit application."""
    st.title("An engine :racing_motorcycle: for refactoring knowledge")
    st.write(
        "Upload a PDF file, extract its content, then process it further with a chosen LLM. :rocket:"
    )

    # Sidebar for API Key Status
    st.sidebar.header("API Key Status")
    if ANTHROPIC_API_KEY:
        st.sidebar.success("Anthropic API Key: Found")
    else:
        st.sidebar.warning("Anthropic API Key: **Missing** (Claude unavailable)")
    if GROQ_API_KEY:
        st.sidebar.success("Groq API Key: Found")
    else:
        st.sidebar.warning("Groq API Key: **Missing** (Groq models unavailable)")
    if OPENAI_API_KEY: 
        st.sidebar.success("OpenAI API Key: Found") 
    else: 
        st.sidebar.warning("OpenAI API Key: **Missing** (OpenAI models unavailable)") 


    prompt_instructions = load_prompts_from_dir(PROMPTS_DIR)
    selected_prompt_name = None
    selected_prompt_instruction = None

    if not PROMPTS_DIR.is_dir():
         st.error(f"Prompts directory not found: {PROMPTS_DIR}. Please create it and add YAML prompt files.")
    elif not prompt_instructions: # Check if dictionary is empty after loading
        st.warning(f"No valid prompt files found or loaded from {PROMPTS_DIR}. Secondary processing with prompts will not be available.")


    if prompt_instructions: # Only show if prompts were loaded
        prompt_names = list(prompt_instructions.keys())
        selected_prompt_name = st.selectbox(
            "Choose a prompt for Secondary Processing",
            options=prompt_names,
            index=0
        )
        selected_prompt_instruction = prompt_instructions[selected_prompt_name]
        with st.expander("Selected Prompt Instruction"):
            st.markdown(f"```\n{selected_prompt_instruction}\n```")
    # else: # Warning is now handled above based on directory and load status
        # st.warning("No prompts loaded. Secondary processing with prompts will not be available.")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    st.sidebar.header("Step 1: Pre-Processing") 
    extraction_method = st.sidebar.radio("Choose PDF Content Extraction Method", ["PyMuPDF", "Claude API"], help="PyMuPDF is local and fast for text. Claude API can interpret content structure more deeply but requires an API key and network.")
    
    claude_pdf_prompt_default = """Extract the full content of the PDF file with the highest possible level of detail.
Include all text, and provide detailed descriptions of any non-text elements such as images, tables, figures, and diagrams.
**Translation:**
* Translate all non-English text content into English.
**Element Descriptions:**
* **Images:** Describe their content, purpose, and any relevant captions or surrounding text. Specify the image's location within the document (e.g., "on page 2, within the 'Results' section").
* **Tables:** Include the table title, column headers, and all data within the table. Preserve the table's original formatting as closely as possible (e.g., using Markdown or HTML). Indicate the table's location within the document.
* **Figures and Diagrams:** Describe their type, content, and any labels or annotations. Specify the figure or diagram's location within the document.
**Output Format:**
Organize the extracted content in a structured format. Clearly separate text from descriptions of non-text elements. For each element, include a heading or label indicating its type (e.g., "Text - Introduction," "Image - Experiment Setup," "Table 1 - Results Summary"). Maintain the original reading order of the document.
"""
    question_for_claude_pdf = st.sidebar.text_area("Question/Instruction for Claude PDF Extraction:", value=claude_pdf_prompt_default, height=150, disabled=(extraction_method != "Claude API"))

    st.sidebar.header("Step 2: Main Text Processing") 
    
    available_processors = []
    if client_anthropic: 
        available_processors.append("Claude")
    if client_groq:
        available_processors.append("Groq")
    if client_openai: 
        available_processors.append("OpenAI") 
    
    if not available_processors:
        st.sidebar.error("No LLMs available for main text processing. Please check API keys.")
        secondary_processor = None
    else:
        secondary_processor = st.sidebar.selectbox(
            "Choose LLM for Main Text Processing:", 
            options=available_processors,
            index=0 if available_processors else -1 
        )
    
    # Groq model selection (user confirmed this list is what they want)
    groq_model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it"] 
    default_groq_model = "llama-3.3-70b-versatile" 
    selected_groq_model = default_groq_model 
    if secondary_processor == "Groq" and client_groq:
        if default_groq_model not in groq_model_options: 
            groq_model_options.insert(0, default_groq_model)
        current_groq_index = 0
        try:
            current_groq_index = groq_model_options.index(default_groq_model)
        except ValueError:
            pass 
        selected_groq_model = st.sidebar.selectbox(
            "Select Groq Model:",
            options=groq_model_options,
            index=current_groq_index 
        )

    # OpenAI model selection & Web Search Toggle
    openai_model_options = ["gpt-4o", "gpt-4.1", "o3", "o4-mini"] 
    default_openai_model = "gpt-4o"
    selected_openai_model = default_openai_model 
    enable_openai_web_search = False 
    web_search_eligible_openai_models = ["gpt-4o", "gpt-4.1"]

    if secondary_processor == "OpenAI" and client_openai:
        if default_openai_model not in openai_model_options:
            openai_model_options.insert(0, default_openai_model)
        current_openai_index = 0
        try:
            current_openai_index = openai_model_options.index(default_openai_model)
        except ValueError:
            pass 
        selected_openai_model = st.sidebar.selectbox(
            "Select OpenAI Model:",
            options=openai_model_options,
            index=current_openai_index
        )
        if selected_openai_model in web_search_eligible_openai_models:
            enable_openai_web_search = st.sidebar.toggle(
                f"Enable Web Search for {selected_openai_model}", 
                value=True, 
                help=f"Uses Responses API with web_search_preview tool for {selected_openai_model}. If disabled, uses Chat Completions API."
            )
        else:
            st.sidebar.caption(f"Web search toggle not applicable for {selected_openai_model}. Uses Chat Completions API.")


    if st.button("🚀 Get Information & Process", disabled=(not uploaded_file or not secondary_processor or not selected_prompt_instruction)):
        if uploaded_file is None: 
            st.error("Please upload a PDF file.")
            return
        if not selected_prompt_instruction:
            st.error("No prompt selected for main text processing.") 
            return
        if not secondary_processor:
            st.error("No main text processor selected or available. Check API Keys.") 
            return
        
        pdf_text = None 
        st.subheader("📄 Step 1: Pre-Processing & PDF Content Extraction") 
        st.info(f"Attempting PDF content extraction with {extraction_method}...") 
        
        uploaded_file.seek(0) 

        if extraction_method == "PyMuPDF":
            pdf_text = read_pdf_with_pymupdf(uploaded_file)
            if pdf_text:
                st.success("Text extracted successfully with PyMuPDF!")
                with st.expander("Extracted Text (PyMuPDF)", expanded=False):
                    st.text_area("PDF Content", pdf_text, height=300, key="pymupdf_output")
            else:
                st.error("Failed to extract text from the PDF using PyMuPDF.")
        elif extraction_method == "Claude API":
            if not client_anthropic: 
                st.error("Anthropic API Key not set or client failed to initialize. Cannot use Claude for PDF extraction.")
            else:
                uploaded_file.seek(0) 
                pdf_text, token_count = read_pdf_with_claude(uploaded_file, question_for_claude_pdf)
                if pdf_text is not None: 
                    st.success(f"Content extracted/processed by Claude. {f'Input Token Count: {token_count}' if token_count else ''}")
                    with st.expander("Content from Claude (PDF Extraction)", expanded=False):
                        st.text_area("Claude's Response", pdf_text, height=300, key="claude_pdf_output")
                else:
                    st.error("Failed to get content from Claude for the PDF. Please check the PDF and question, and try again.")
        
        if pdf_text and selected_prompt_instruction:
            st.subheader(f"⚙️ Step 2: Main Text Processing with {secondary_processor}") 
            primary_output, secondary_output = None, None 
            
            if secondary_processor == "Claude":
                if not client_anthropic:
                    st.error("Anthropic client not available for main text processing.") 
                else:
                    primary_output, secondary_output = process_text_with_claude(pdf_text, selected_prompt_instruction)
            elif secondary_processor == "Groq":
                if not client_groq:
                    st.error("Groq client not available for main text processing.") 
                else:
                    primary_output, secondary_output = process_text_with_groq(pdf_text, selected_prompt_instruction, model_name=selected_groq_model)
            elif secondary_processor == "OpenAI": 
                if not client_openai:
                    st.error("OpenAI client not available for main text processing.") 
                else:
                    primary_output, secondary_output = process_text_with_openai(
                        pdf_text, 
                        selected_prompt_instruction, 
                        model_name=selected_openai_model,
                        enable_web_search=enable_openai_web_search 
                    )

            if primary_output: 
                if secondary_processor == "Claude" and secondary_output: 
                    with st.expander("Working/Intermediate Output (Claude)", expanded=True):
                        st.markdown(secondary_output) 
                        st_copy_to_clipboard(secondary_output, key="copy_working_claude")
                
                with st.expander("Final Outcome", expanded=True):
                    st.markdown(primary_output) 
                    st_copy_to_clipboard(primary_output, key="copy_outcome")

                if secondary_processor == "OpenAI" and secondary_output: 
                     with st.expander("Sources (OpenAI)", expanded=True):
                        st.markdown(secondary_output) 
                        st_copy_to_clipboard(secondary_output, key="copy_sources_openai")
                
                st.balloons() 
            elif pdf_text: 
                 st.error(f"Main text processing with {secondary_processor} did not produce an outcome. API key might be missing or an error occurred.") 

        elif not pdf_text:
            st.error("PDF text extraction failed in Step 1 (Pre-Processing), so Step 2 (Main Text Processing) was skipped.") 
        elif not selected_prompt_instruction: 
            st.warning("No prompt selected for main text processing. Step 2 was skipped.") 


def main():
    st.set_page_config(page_title="Information Alchemist", page_icon=":rocket:", layout="wide")
    # Password check
    if not check_password():
        st.stop()  # Do not run the rest of the app if the password is not correct

    main_app_content() # Run the main app


if __name__ == "__main__":
    main()
