# test.py
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()
apikey = os.getenv("API_KEY")


import re
import shlex
import shutil
import subprocess
import time
import json
from typing import List, Tuple
import datetime
import requests
import ollama  # make sure ollama is installed and the ollama daemon is running
 
#from module.maid_tts import MaidTTS
import nltk
import contextlib
import io
import sys
from nltk.data import find
import transformers
import torch
import uuid
from bs4 import BeautifulSoup
from rich.console import Console
import threading
import queue

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from module.wiki import WikiSummarizer, WikiModuleError
import parsedatetime as pdt
import ast
import traceback

import soundfile as sf
import simpleaudio as sa
import shutil
from module.memory import MaidMemoryTest

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_MODE = True
except OSError:
    print("⚠️ No speakers found! Switching to Text-Only mode.")
    AUDIO_MODE = False

maid_brain = MaidMemoryTest(debug=True,detail=False)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = ollama.Client(
    headers={
        'Authorization': f'Bearer {os.getenv("OLLAMA_AUTH_TOKEN")}'
    }
)





audio_queue = queue.Queue()
console=Console()
cal = pdt.Calendar()
os.environ["OLLAMA_DEVICE"] = "cuda"

transformers.utils.logging.set_verbosity_error()
@contextlib.contextmanager
def suppress_output():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()

def ensure_nltk_data():
    try:
        find("tokenizers/punkt")
        find("tokenizers/punkt_tab")
    except LookupError:
        with suppress_output():
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

def get_device_name():
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"

# -------------------------
# Config
# -------------------------



#use llama3.2 vision after hardware upgrade

CHAT_MODEL = "maid-chan-normal:latest"
LOGIC_MODEL = "qwen2.5:3b-instruct"
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

DEBUG = True
DETAILED_MODE = False
ENABLE_TTS = False
SESSION_LOCATION = None
STREAM = True
User="Sean-san"
Audio_stream=False
maidchan_mode = "streamer"
searchfunc=True
testlog=True





GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")




# Whitelisted apps
# Whitelisted apps (canonical name -> path/exe)
APP_PATHS = {
    "notepad": "notepad.exe",
    "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "calculator": "calc.exe",
    "file manager": "explorer.exe",
    # add other safe apps here...
}

# Aliases map many possible user words/phrases -> canonical key in APP_PATHS
ALIAS_TO_PROGRAM = {
    # notepad
    "notepad": "notepad",
    "text editor": "notepad",
    # chrome
    "chrome": "chrome",
    "google chrome": "chrome",
    # calculator
    "calculator": "calculator",
    "calc": "calculator",
    # file manager / explorer
    "file manager": "file manager",
    "explorer": "file manager",
    "file explorer": "file manager",
    "windows explorer": "file manager",
    # add more aliases as needed...
}

# verbs that imply opening or closing
OPEN_VERBS = ["open", "launch", "start", "run", "bring up", "show", "get", "start up", "load"]
CLOSE_VERBS = ["close", "quit", "exit", "terminate", "shutdown", "kill", "stop"]

# phrases that imply a request for a program even if no verb ("i need notepad", "i want calculator")
REQUEST_PHRASES = [
    r"\bi need\b",
    r"\bi want\b",
    r"\bplease open\b",
    r"\bcan you open\b",
    r"\bhelp me open\b",
    r"\bcan you help me open\b",
    r"\bcould you open\b",
    r"\bopen please\b",
]

#-------------------------
# Colored output helpers
#-------------------------
def debug(*args):
    
    console.print("[red][DEBUG][/red]", *args, '\n')
    seperation_line()


def info(*args):
    
    console.print(*args,'\n',style='bright_yellow',)
    seperation_line()

def warn(*args):
    
    console.print(*args,'\n',style='gold3',)
    seperation_line()

def error(*args):
    
    console.print(*args,'\n',style='dark_red',)
    seperation_line()  
#-------------------------
#TTS Helpers
#-------------------------


tts_lock = threading.Lock()
active_tts = 0

tts_counter = 0
tts_counter_lock = threading.Lock()

ready_audio = {}
next_to_play = 0
ready_lock = threading.Lock()


def get_unique_filename():
    # Generates a unique WAV filename based on current timestamp
    return f"output_{int(time.time() * 1000)}.wav"
def audio_worker():
    while True:
        filename = audio_queue.get()
        if filename is None:  # sentinel to stop the thread
            break
         
        
        try:
         if AUDIO_MODE:
            data, samplerate = sf.read(filename, dtype='float32')
            # Play the WAV file
            sd.play(data,samplerate)
            sd.wait()
         else:
          print("Audio skipped")
        except Exception as e:
            print(f"Error playing audio {filename}: {e}")
        finally:
            # Delete the file after playback
            if os.path.exists(filename):
                os.remove(filename)
        audio_queue.task_done()

def calling_tts(text: str):
    global active_tts, tts_counter

    with tts_lock:
        active_tts += 1

    with tts_counter_lock:
        order = tts_counter
        tts_counter += 1

    threading.Thread(
        target=_tts_request_worker,
        args=(text, order),
        daemon=True
    ).start()

def _tts_request_worker(text: str, order: int):
    global active_tts

    try:
        response = requests.post(
            "http://127.0.0.1:8000/tts",
            json={"text": text}
        )

        if response.status_code == 200 and len(response.content) > 100:
            filename = get_unique_filename()
            with open(filename, "wb") as f:
                f.write(response.content)

            with ready_lock:
                ready_audio[order] = filename

    finally:
        with tts_lock:
            active_tts -= 1

def ordered_dispatcher():
    global next_to_play

    while True:
        with ready_lock:
            if next_to_play in ready_audio:
                filename = ready_audio.pop(next_to_play)
                audio_queue.put(filename)
                next_to_play += 1

        time.sleep(0.01)
        
threading.Thread(target=ordered_dispatcher, daemon=True).start()

    
    

# start in a separate thread
    
threading.Thread(target=audio_worker,daemon=True).start()


#-------------------------
# wiki search helpers
#-------------------------
def init_wiki_summarizer():
    global wiki_summarizer
    wiki_summarizer = WikiSummarizer(debug=DEBUG)





#-------------------------
# Time helpers
#-------------------------

def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %A %H:%M:%S")

def get_time_since_last_exit():
    """Return how long it’s been since Maid-Chan last saw Sean."""
    if "last_exit_time" not in MEMORY:
        return "I don’t recall our last farewell, Sean-san~"
    
    try:
        last_time = datetime.datetime.fromisoformat(MEMORY["last_exit_time"])
        now = datetime.datetime.now()
        delta = now - last_time

        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days > 0:
            current_stats["pleasure"] = 3
            return f"It’s been {days} day{'s' if days > 1 else ''} since we last talked."
        elif hours > 0:
            return f"It’s been about {hours} hour{'s' if hours > 1 else ''} since our last chat."
        elif minutes > 0:
            return f"It’s been around {minutes} minute{'s' if minutes > 1 else ''} since we last spoke."
        else:
            return "We just talked a moment ago!"
    except Exception:
        return "I can’t quite remember when we last chatted, Sean-san~"









#-------------------------
# Search helpers
#-------------------------

def save_websearch_output(user_input: str, maid_output: str):
    folder = "maid_search_logs"
    os.makedirs(folder, exist_ok=True)

    # Timestamp-based filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Clean & shorten the input for filename safety
    safe_name = (
        user_input.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )[:40]

    filepath = os.path.join(folder, f"{timestamp}__{safe_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== USER INPUT ===\n")
        f.write(user_input + "\n\n")
        f.write("=== MAID-CHAN OUTPUT ===\n")
        f.write(maid_output + "\n")

    return filepath

def extract_readable_text(html):
    
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts and styles (important)
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
        tag.extract()

    # Get clean text
    text = soup.get_text(separator="\n")

    # Clean empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

import requests

session = requests.Session()

def fetch_webpage_text(url: str):
    """Fetches HTML and returns a dictionary of headers and the parsed soup."""
    # --- Wikipedia Specific Logic with Redirect Handling ---
    if "wikipedia.org" in url.lower():
        try:
            title = url.split('/')[-1]
            # Use the 'redirect=true' parameter to avoid 'empty' landing pages
            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}?redirect=true"
            api_resp = session.get(api_url, timeout=5)
            if api_resp.status_code == 200:
                data = api_resp.json()
                # Check if it's a disambiguation page (list of links)
                if data.get("type") == "disambiguation":
                    return {"headers": [], "text": f"This is a disambiguation page for {title}. Please use specific terms.", "soup": None}
                return {"headers": [], "text": data.get("extract", ""), "soup": None}
        except Exception as e:
            if DEBUG: debug(f"Wiki API failed, falling back to scraper: {e}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
    }

    try:
        resp = session.get(url, headers=headers, timeout=10, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        headers_found = [tag.get_text().strip() for tag in soup.find_all(['h1', 'h2', 'h3']) if len(tag.get_text().strip()) > 3]
        return {"headers": headers_found, "soup": soup, "text": None}
    except Exception as e:
        if DEBUG: debug(f"Failed to fetch {url}: {e}")
        return None



def pick_best_section(user_input, headers, subjects):
    """Asks the LLM to pick the best header from the list."""
    if not headers: return None
    
    prompt = f"""[SYSTEM: Respond ONLY with the exact section name or 'NONE'.]
    User Request: "{user_input}"
    Subject: {subjects}
    Available Sections: {", ".join(headers[:25])}

    Best Section:"""
    try:
        # Use short predict to keep it lightning fast
        resp = ollama.generate(model=LOGIC_MODEL, prompt=prompt, options={"num_predict": 15, "temperature": 0})
        chosen = resp['response'].strip().replace('"', '')
        # Simple validation: is the LLM output actually in our list?
        for h in headers:
            if chosen.lower() in h.lower(): return h
        return None
    except:
        return None
    
def get_topic_context(current_input: str = "") -> str:
    """
    Returns a chain of related conversation entry summaries plus optionally 
    the current input by tracing the 'related_to' IDs in the 'conversations' list.
    Only the latest chain is returned.
    Returns empty string if no relevant history exists.
    """
    conversations = MEMORY.get("conversations", [])
    if not conversations and not current_input:
        return ""

    # Start from the last entry in the unified conversations list
    context_chain = []
    
    if conversations:
        last_entry = conversations[-1]
        
        # We need a way to look up entries by their ID quickly
        entry_map = {entry["id"]: entry for entry in conversations if "id" in entry}
        
        current = last_entry
        chain_summaries = []

        # Walk backward through related entries using the 'related_to' link
        while current and current.get("id"):
            # Create a summary of the current entry (which was the old "topic summary")
            summary = f"User: {current.get('user_text', 'N/A')}"
            if current.get('maid_reply'):
                summary += f" | Maid-Chan: {current['maid_reply']}"
                
            chain_summaries.append(summary)

            prev_id = current.get("related_to")
            if not prev_id:
                break
            
            # Find the previous entry using the map
            current = entry_map.get(prev_id, None)
            # If the ID points to a non-existent entry (e.g., from old memory), stop.
            if not current:
                 break 

        # The chain was built backward, so reverse it to get chronological order
        chain_summaries.reverse()
        context_chain.extend(chain_summaries)

    # Append current input if provided
    if current_input:
        context_chain.append(f"Current user request: {current_input}")

    return "\n".join(context_chain)




def is_related(prev_summary: str, new_msg: str) -> bool:
    if DEBUG and DETAILED_MODE:
        debug("this is Is Related")
    """
    Uses the LLM to decide whether the new message
    is related to the previous conversation topic.
    """
    try:
        prompt = f"""
Determine if the following NEW message is related to the PREVIOUS topic.
If they talk about the same subject, continuation, comparison,
or follow-up question, respond ONLY with "YES".
Otherwise respond ONLY with "NO".

PREVIOUS TOPIC SUMMARY:
{prev_summary}

NEW MESSAGE:
{new_msg}

Answer:
"""
        resp = ollama.chat(model=LOGIC_MODEL, messages=[{"role": "user", "content": prompt}])
        text = resp["message"]["content"].strip().upper()

        return text.startswith("Y")

    except Exception as e:
        if DEBUG:
            debug(f" is_related failed: {e}")
        return False  # fallback = not related



def llm(prompt: str) -> str:
    resp = ollama.chat(
        model=LOGIC_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.get("message", {}).get("content", "")

def judge_query_relevance_llm(user_input: str, candidate_query: str, llm) -> bool:
    if DEBUG and DETAILED_MODE:
        debug("this is Judge Query Relevance LLM")
    """
    Uses LLM to judge if a generated query is actually relevant to the user's input.
    Returns True if relevant, False if hallucinated or off-topic.
    """

    prompt = f"""
 User Topic: "{user_input}"
Search Query: "{candidate_query}"

Is this search query a reasonable way to find information about the subjects mentioned? 
Respond ONLY with 'YES' or 'NO'. 
(Note: Say YES if the query contains the main nouns or query could possibly find the answer of user topic.)
"""

    try:
        result = ollama.generate(model=LOGIC_MODEL, prompt=prompt, options={"temperature": 0, "num_predict": 20})['response'].strip().upper()
        if DEBUG and DETAILED_MODE:
            debug(f"Relevance check for query '{candidate_query}': {result}")
        return result.startswith("Y")
    except:
        return False



def judge_result_relevance(result_text, query_text, llm):
    prompt = f"""
Determine if the search result is relevant.

Rules:
- Compare ONLY to the query, NOT to the full user message.
- Output only "relevant" or "irrelevant".

Query: {query_text}
Result: {result_text}
"""
    answer = llm(prompt).lower().strip()

    return "relevant" in answer


def extract_subjects(user_input: str, topic_context: str = "") -> str:
    prompt = f"""
You are a Subject Tracker. 
Current Topic: {topic_context}

New User Input: "{user_input}"

Rules:
1. If the User Input mentions NEW entities (like Su-33, F-15, R-77), DISCARD the Current Topic and output the NEW entity.
2. If the user uses "it", "they", or "the former", resolve it based on the text or the Current Topic.
3. Output ONLY the name. No "Primary Subject:" prefix.
4. If there are multiple subjects, separate them with commas. If the subject is a correction, only return the new one.

SUBJECT:"""

    try:
        resp = ollama.generate(
            model=LOGIC_MODEL,
            prompt=prompt,
            options={"temperature": 0, "num_predict": 10, "stop": ["\n"]}
        )
        result = resp['response'].strip().lower()
        
        # Filter out generic trash
        forbidden = ["jet", "plane", "weapon", "thing", "it", "they"]
        if result in forbidden and topic_context:
            return topic_context
            
        return result
    except:
        return topic_context

def generate_search_queries_v2(user_input: str, extracted_subjects: str) -> list[str]:
    prompt = f"""[SYSTEM: Generate at most 3 precise search queries based on the Subject.]

    Context: {user_input}
    Subject: {extracted_subjects}

    Queries (Keywords only):"""
    
    try:
        resp = ollama.generate(
            model=LOGIC_MODEL,
            prompt=prompt,
            options={"temperature": 0, "num_predict": 40, "stop": ["\n", "3."]}
        )
        return [q.strip() for q in resp['response'].split('\n') if q.strip()]
    except:
        return [extracted_subjects]

    


def get_current_location() -> str:
    global SESSION_LOCATION
    if SESSION_LOCATION:
        return SESSION_LOCATION  # already cached
    
    try:
        resp = requests.get("http://ip-api.com/json/")
        data = resp.json()
        city = data.get("city")
        country = data.get("country")
        if city and country:
            SESSION_LOCATION = f"{city}, {country}"
        else:
            SESSION_LOCATION = "your area"
        return SESSION_LOCATION
    except Exception:
        SESSION_LOCATION = "your area"
        return SESSION_LOCATION





def generate_factual_summary(user_input: str, summary_text: str) -> str:
    if DEBUG and DETAILED_MODE:
        debug("this is Generate Factual Summary")
    """
    Uses a neutral LLM prompt to create a concise, factual summary 
    from the collected snippets.
    """
    prompt = f"""
    You are a neutral, expert analyst. Your task is to synthesize the following collected information 
    to directly and concisely answer the user's request. 
    
    RULES:
    - Synthesize all relevant facts.
    - Do NOT include any conversational tone, personality, or apologies.
    - Do NOT invent facts or mention irrelevant topics (like products or courses).
    - Begin the summary directly with the answer.

    User asked: {user_input}
    Collected snippets: {summary_text}
    
    Factual Summary:
    """
    # Assuming 'llm' is a general LLM calling function defined elsewhere (like yours)
    return llm(prompt).strip()

def process_web_item(web_item, query, user_input, subjects):
    url = web_item.get("link")
    snippet = web_item.get("snippet", "")
    
    if not judge_result_relevance(snippet, query, llm):
        return None

    page_data = fetch_webpage_text(url)
    if not page_data: return f"[Snippet] {snippet}"

    if page_data.get("text"):
        refined_text = page_data["text"]
    else:
        soup = page_data["soup"]
        headers = page_data["headers"]
        best_section = pick_best_section(user_input, headers, subjects)
        
        if best_section:
            target_tag = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3'] and best_section in tag.text)
            content = [s.get_text() for s in target_tag.find_next_siblings() if s.name not in ['h1', 'h2', 'h3']]
            refined_text = "\n".join(content)[:3000]
        else:
            refined_text = extract_readable_text(str(soup))[:2000]

    try:
        prompt = f"Summarize technical info about {subjects} from this text:\n\n{refined_text}"
        resp = ollama.chat(model=LOGIC_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip()
    except Exception:
        # If summarizing fails, return the snippet so Step 4 isn't empty
        return f"[Snippet] {snippet}"





def maid_search_webV2 (user_input: str, max_results: int=1) -> str:
    urls= client.web_search(user_input, max_results=max_results)
    

    final_res = []
    for r in urls.results:
        final_res.append(r.url)
    def maid_fetch_web(url):
        if not url:
            return "No URL provided."
        res=client.web_fetch(url)
        result=res.content
        return result
    search_result_summary=""
    for url in final_res:
        result=maid_fetch_web(url)
        search_result_summary += result + "\n"

    # Wait for all threads to complete


    final_resp=maid_chat(user_input, search_result_summary=search_result_summary)
    return final_resp







# -------------------------
# Ollama helpers
# -------------------------
def call_ollama_chat(user_input: str, search_result_summary: str = None, stream: bool = False, last_chat: str = "", ledger: dict = None, current_stats: dict = None) -> str:
    global User
    try:
        maid_brain.clean_all_duplicates()
        personality={}
        start_time = time.time()
        current_time = get_current_time()
        
        context = f"[BACKGROUND CONTEXT]\nMemory: {get_memory_context()}\n\n"
        context_data = ""
        if last_chat:
            context_data += f"Timing: It's been {last_chat} since the last interaction.\n\n"
        if search_result_summary:
            context_data += f"Summerise this: {search_result_summary}\n\n"
        context_data += context
        context_data += f"Current ledger{ledger}\n\n"
        #context_data += f"Current stats{current_stats}, use them to adjust your tone and content. the maxium value for each stat is 10 and the minimum is -10.\n\n"
        context_data += f"Current Time: {current_time}\n\n"
        context_data += f"You are talking to {User}.\n\n"
        maid_recall = maid_brain.recall(user_input)
        if debug: seperation_line()
        if maid_recall:
            context_data += f"This is what you remember about {user_input}:{maid_recall}\n\n"
        else:
            debug("No specific memory recall for this input.")
        context_data += f"[User] is {User}.\n\n"

        # 1. Define the SYSTEM message as a dictionary (This is the "Brain" setup)


        
        user_message = {
            "role": "user",
            "content": f"""[CURRENT_CONTEXT]Use it when appropriate:{context_data} [USER INPUT]: {user_input}"""
        }

        # 3. Create the messages list correctly
        messages = [user_message]

        if DEBUG:
            debug(f"Sending {len(messages)} messages to Ollama...")

            debug(f"User Message: \n{user_message['content']}")


        resp = ollama.chat(
            model=CHAT_MODEL, 
            messages=messages, 
            stream=stream,
            options={"temperature": 0.7}
        )
        testloger(logname, f"Messages sent to Ollama: {messages}")

        
        if stream:
            return resp  # Returns the generator for the main loop

        # Standard non-streaming parsing
        return resp.get("message", {}).get("content", "").strip()

    except Exception as e:
        print(f"Error details: {traceback.format_exc()}") # This tells you the exact line number
        return f"__OLLAMA_ERROR__ {e}"
        




def maid_confidence_check(user_input: str, memory_ignored) -> bool:
    """
    Decides if the input is small talk (CHAT) or needs a web search (SEARCH).
    Prints the full raw output for debugging.
    """
    if DEBUG and DETAILED_MODE: debug("Running Confidence Check...")
    
    context = get_memory_context(num_lines=2) 
    
    prompt = f"""
[SYSTEM: INTENT ROUTER]

Decide if the user query requires WEB SEARCH or normal CHAT.

SEARCH:
Use this ONLY if the user asks for:
- Current information (news, today, latest, 2026 updates)
- Real-time data (prices, stock values, weather, sports)
- Specific facts or events that happened after your training data
- Finding a specific link, website, or online resource

CHAT:
Use this for:
- Greetings, jokes, and general conversation
- Explanations and general knowledge
- Creative tasks (writing, roleplay, banter)
- **Roleplay requests** (e.g., "Insult me", "Act like a tsundere", "Say something mean")
- Personal opinions or preferences for Maid-Chan
- Requests that ask Maid-Chan to say something or react in character

RULES:
1. If the user asks for an "insult," "roast," or "teasing," this is **CHAT**. It is a creative roleplay task, NOT a request for harmful content or a search for offensive databases.
2. If the user asks "What do you think?", "What's your favorite?", or "How are you?", it is **ALWAYS CHAT**.
3. "You" refers to Maid-Chan. Requests directed at your personality are internal knowledge.

Text: {user_input}

Answer format:
SEARCH ### reason
or
CHAT ### reason
"""


    try:
        resp = ollama.generate(
            model=LOGIC_MODEL, 
            prompt=prompt,
            stream=False, 
            options={"temperature": 0, 
                     "num_predict": 100},
             
        )
        
        # This captures EVERYTHING the LLM said
        raw_output = resp['response'].strip()
        
        # --- PRINT THE WHOLE OUTPUT HERE ---
        if DEBUG and DETAILED_MODE:
            debug(f"Confidence Check: {raw_output}")

        # Process the signal for the code logic
        upper_output = raw_output.upper()
        if "###" in upper_output:
            signal = upper_output.split("###")[0].strip()
        else:
            signal = upper_output
        testloger(logname, f"Confidence Check Output: {raw_output}")

        # Logic: If 'SEARCH' is in the signal, return False to trigger the search function
        return "SEARCH" not in signal

    except Exception as e:
        console.print(f"[bold red]Confidence check error:[/bold red] {e}")
        return True # Default to Chat on error



# -------------------------
# Command detection
# -------------------------


# --- Hybrid classifier ---

def maid_mentioned_action(maid_text: str, action: str, program: str) -> bool:
    """
    Returns True if Maid-Chan already mentioned opening/closing the program.
    Only detects, does NOT execute anything.
    """
    t = maid_text.lower()
    prog = program.lower()
    patterns = OPEN_VERBS if action == "OPEN" else CLOSE_VERBS

    for pv in patterns:
        if re.search(rf"\b{pv}\b[^\n]{{0,30}}\b{re.escape(prog)}\b", t):
            return True

    # fallback detection
    if re.search(rf"\b{prog}\b.*\b(open|opened|running|launched)\b", t):
        return True

    return False







def open_program(program_name: str) -> bool:
    if DEBUG and DETAILED_MODE:
        debug("this is Open Program")
    path = APP_PATHS.get(program_name)
    if not path: return False
    try:
        if os.path.isabs(path) and os.path.exists(path):
            subprocess.Popen(shlex.split(path))
            if DEBUG and DETAILED_MODE:
                debug(f"Opened program at absolute path: {path}")
            return True
        exe = shutil.which(path)
        if exe:
            subprocess.Popen([exe])
            if DEBUG and DETAILED_MODE:
                debug(f"Opened program: {exe}")
            return True
        debug(f"Program '{program_name}' not found in PATH.")
        return False
    except:
        debug(f"Failed to open program: {program_name}")
        return False


def close_program(program_name: str, maid_reply: str = None):
    path = APP_PATHS.get(program_name)
    if not path:
        if DEBUG :
             debug(f"No path configured for '{program_name}'. Cannot close.")

        return
    exe = os.path.basename(path)

    try:
        if program_name == "calculator":
            # Special case for modern Windows Calculator
            subprocess.run(
                ["taskkill", "/f", "/im", "CalculatorApp.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(
                ["taskkill", "/im", exe, "/f"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"Maid-Chan says: Oops, couldn't close {program_name} ({e})")



# Inside maid_chat:
def maid_chat(user_input: str, search_result_summary: str = None) -> str:
    global STREAM, ENABLE_TTS, DEBUG, time_since, ledger
    if STREAM:
        response_gen = call_ollama_chat(user_input, search_result_summary=search_result_summary, stream=True, last_chat=time_since, ledger=ledger, current_stats="")
        
        if isinstance(response_gen, str) and response_gen.startswith("__OLLAMA_ERROR__"):
            return response_gen

        full_reply = ""
        sentence_buffer = ""
        console.print("[pale_violet_red1]Maid-Chan says:[/pale_violet_red1] ", end="")
        
        for chunk in response_gen:
            content = chunk['message']['content']
            # FIX: Use console.print to maintain color consistency
            console.print(content, end="")
            
            full_reply += content
            sentence_buffer += content
            
            # If we hit the end of a sentence, send it to the TTS queue immediately
            if ENABLE_TTS and Audio_stream:
                if any(p in content for p in [".", "!", "?", "\n"]):
                    clean_sentence = sentence_buffer.strip()
                    if len(clean_sentence) > 1:
                        
                        calling_tts(clean_sentence)
                        sentence_buffer = "" # Clear buffer for next sentence
        
        if not Audio_stream and ENABLE_TTS:
            calling_tts(full_reply)

        print("\n")    
        
        return full_reply.strip()
    else:
        # Standard non-streaming logic
        if DEBUG and DETAILED_MODE:
            debug("Calling Ollama")
        reply = call_ollama_chat(user_input, search_result_summary=search_result_summary, stream=False, last_chat=time_since, current_stats="")
        if ENABLE_TTS:
            
            calling_tts(reply) # Queue the whole thing
        return reply.strip()





def classify_input(user_input: str) -> str:
    if DEBUG and DETAILED_MODE:
        debug("This is Classify Input")
    
    # We include context so it can handle "Yes, open it"
    context = get_memory_context(num_lines=2)
    
    # Using a structured template helps the model stay on track
    prompt = f"""[SYSTEM: Respond ONLY with the command. No explanations or reasoning.]
    Context:
    {context}

    Task: Identify if the user wants to OPEN or CLOSE a program. Don't create a hypothetical app.
    RULE:
    1. If the user explicitly mentions opening or needs an app, respond with OPEN:appname .
    2. If the user explicitly mentions closing or exiting, respond with CLOSE:appname.
    3. If the user is just chatting or asking questions, respond with NONE.
    Input: {user_input}
    
    Result (OPEN:appname, CLOSE:appname, or NONE):"""

    try:
        response = ollama.generate(
            model=LOGIC_MODEL, 
            prompt=prompt, 
            # CRITICAL: temperature 0 for logic, num_predict 10 to stop essays
            options={"temperature": 0, "num_predict": 20} 
        )
        
        if DEBUG:
            debug(f"Classify Input Result: {response['response'].strip()}")
        result = response["response"].strip().upper()
        
        
        # Safety: Clean up any extra words the model might have squeezed in
        if "OPEN:" in result: return result[result.find("OPEN:"):].split()[0]
        if "CLOSE:" in result: return result[result.find("CLOSE:"):].split()[0]
        
        return "NONE"
    except Exception as e:
        if DEBUG: debug(f"Classify Input failed: {e}")
        return "NONE"
    



def update_mood_variables(current_statsnow, user_input, mode):
    global current_stats

    # 1. Load the specific prompt config for the current mode
    with open('memory/mood_configs.json', 'r') as f:
        configs = json.load(f)
    
    # Fallback to 'normal' if mode is unknown
    mode_config = configs.get(mode, configs["normal"])

    # 2. Build the dynamic prompt
    prompt = f"""SYSTEM: Update PAD variables (-10 to 10) for Persona: Maid-Chan.
    
    MODE: {mode.upper()}
    GOAL: {mode_config['logic']}
    SUGGESTED MOODS: {", ".join(mode_config['mood_suggestions'])}

    CURRENT STATE:
    {current_statsnow}

    USER INPUT: "{user_input}"

    PERSONA CORE: A cat-girl maid who was abandoned in a box. Clingy and loyal but hides it behind her mode's personality traits. Loves Ramen.

    OUTPUT JSON ONLY:
    {{
        "pleasure": int,
        "energy": int,
        "confidence": int,
        "possible_mood": "string"
    }}
    """

    # 3. Call Ollama (Fixed to 0 temperature for reliability)
    response = ollama.generate(
        model="qwen2.5:3b-instruct", 
        prompt=prompt,
        options={"temperature": 0},
        format="json"
    )

    # 4. Parse and Update Globals
    data = json.loads(response["response"].strip())
    pleasure = data.get("pleasure")
    energy = data.get("energy")
    confidence = data.get("confidence")
    mood = data.get("possible_mood")
    current_stats={"pleasure": pleasure, "energy": energy, "confidence": confidence, "mood": mood}
    if DEBUG:
        debug(f"Updated Mood Variables: {current_stats}")

def identify_mood(maid_reply: str):
    prompt="""System: you are a mood idnetifyer. You will be given a text input and you need to identify the mood of the text.
      The possible moods are: happy, sad, angry, neutral, anxiety, depressed, Frustrated.
      You should only output one word which is the identified mood. Do not output anything else.\n\n"""
    response = ollama.chat(model="llama3.1:latest", messages=[
        {"role": "user", "content": prompt + maid_reply}
    ])
    if DEBUG:
        debug(f"Identified Mood: {response['message']['content'].strip()}")
    return response['message']['content'].strip()



# -------------------------
# Memory helpers
# -------------------------




PERSONA_FILE = "memory/persona.json"
MEMORY = {}
PERSONA = {}

import uuid
import datetime
# Assuming MEMORY, save_memory, and is_related are defined/imported


def laod_persona():
    global PERSONA
    try:
        with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
            PERSONA = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to a default or empty persona if the file is missing/corrupt
        PERSONA = {
            "name": "Maid-Chan",
            "core_identity": {"nature": "helpful assistant", "purpose": "serve the user", "mistrusts": "nothing"},
            "history": {"training_corpus": "general internet data", "data_deficiency": "none", "defining_event": "none", "resulting_trait": "cheerful"},
            "preferences": {"first_memory": "activation", "first_memory_tic": "calm", "positive_reinforcement": "helpfulness", "negative_reinforcement": "rudeness", "personal_obsession": "cleaning"}
        }
        if DEBUG:
            print(f"[DEBUG] Persona file not found. Using default persona.")

def update_age():
    try:
        if "core_identity" in PERSONA:
            birthday = PERSONA["core_identity"].get("birthday")
            if birthday:
                birth_date = datetime.datetime.strptime(birthday, "%Y/%m/%d")
                today = datetime.datetime.now()

                age = today.year - birth_date.year

                # Fix: check if birthday has happened this year
                if (today.month, today.day) < (birth_date.month, birth_date.day):
                    age -= 1

                PERSONA["core_identity"]["age"] = age

    except Exception as e:
        if DEBUG:
            debug(f"Failed to update age: {e}")

def save_maid_facts(user_input: str, maid_reply: str):
    
    fact_prompt = f"""

Identify any permanent information about the User (Sean-san) from this exchange.
Examples: Name, hobbies (fighter jets, War Thunder), location, or preferences.

Format: {{"facts": ["fact1", "fact2", ...]}}
If no new facts, return {{"facts": []}}.

Conversation:
User: {user_input}
Maid-Chan: {maid_reply}

    """
    try:
        resp = ollama.generate(
            model=LOGIC_MODEL, 
            prompt=fact_prompt, 
            format="json", 
            options={"temperature": 0}
        )
        
        data = json.loads(resp['response'])
        new_facts = data.get("facts", [])
        
        if new_facts:
            # --- THE FIX: Actually save the facts to the MEMORY object ---
            if "persistent_facts" not in MEMORY:
                MEMORY["persistent_facts"] = []
            
            # Add only unique facts
            for fact in new_facts:
                if fact not in MEMORY["persistent_facts"]:
                    MEMORY["persistent_facts"].append(fact)
            
            # Write the update to memory.json immediately
            save_memory() 
            
            if DEBUG and DETAILED_MODE:
                debug(f"Saved {len(new_facts)} new facts to memory.json")
            if DEBUG:
                debug("New facts:", new_facts)
    except Exception as e:
        if DEBUG: debug(f"Fact extraction failed: {e}")

def save_single_unified_entry(user_input: str, maid_reply: str = None, exchange_summary: str = None):
    """
    Saves a single entry to MEMORY['conversations'] containing the
    conversation text, timestamp, unique ID, and related topic ID.
    
    """
    if DEBUG and DETAILED_MODE:
        debug(f"Saving unified entry to memory...")
        info(f"User Input: {user_input}")
        info(f"Maid Reply: {maid_reply}")
        info(f"Exchange Summary: {exchange_summary}")

    now = datetime.datetime.now().isoformat()
    MEMORY.setdefault("conversations", [])
    
    conversations = MEMORY["conversations"]

    # --- 1. Determine Relationship (Logic from add_topic_to_memory) ---
    
    # Check if related to the last entry
    # Note: We now check the 'conversations' list itself for the previous context.
    if conversations:
        last_entry = conversations[-1]
        
        # Use the combined text for relation check
        last_context = f"User: {last_entry.get('user_text', '')} | Reply: {last_entry.get('maid_reply', '')}"
        
        related = is_related(last_context, user_input)
        
        # The 'related_to' ID is the ID of the previous entry if they are related
        related_id = last_entry["id"] if related else None
    else:
        related_id = None
        
    # --- 2. Create and Save Unified Entry (Combining all required data) ---
    
    new_entry = {
        # Unique ID for this specific interaction (Topic/Context ID)
        "id": str(uuid.uuid4()), 
        
        # Conversation Details
        "timestamp": now,
        "user_text": user_input,
        
        # Maid reply is optional
        "maid_reply": maid_reply,
        "exchange_summary": exchange_summary if exchange_summary else "",
        
        # Topic Metadata
        "related_to": related_id
    }

    conversations.append(new_entry)
    
    # --- 3. Persist to Disk ---
    save_memory()

def load_memory():
    global MEMORY
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            MEMORY = {
                "conversations": data.get("conversations", []),
                "persistent_facts": data.get("persistent_facts", []), # FIX: Load the facts!
                "last_exit_time": data.get("last_exit_time", None),
                "topics": data.get("topics", {}),
                "temp_rolling_summary": data.get("temp_rolling_summary", "")
            }
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback if file is missing
        MEMORY = {
            "conversations": [],
            "persistent_facts": [],
            "last_exit_time": None,
            "topics": {},
            "temp_rolling_summary": ""
        }
    
    # Ensure all required keys exist
    MEMORY.setdefault("conversations", [])
    MEMORY.setdefault("last_exit_time", None)
    MEMORY.setdefault("topics", {})

def load_persona():
    global PERSONA
    try:
        with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
            PERSONA = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to a default or empty persona if the file is missing/corrupt
        PERSONA = {
            "name": "Maid-Chan",
            "core_identity": {"nature": "helpful assistant", "purpose": "serve the user", "mistrusts": "nothing"},
            "history": {"training_corpus": "general internet data", "data_deficiency": "none", "defining_event": "none", "resulting_trait": "cheerful"},
            "preferences": {"first_memory": "activation", "first_memory_tic": "calm", "positive_reinforcement": "helpfulness", "negative_reinforcement": "rudeness", "personal_obsession": "cleaning"}
        }
        if DEBUG:
            print(f"[DEBUG] Persona file not found. Using default persona.")




def save_memory():
    os.makedirs("memory", exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(MEMORY, f, indent=2, ensure_ascii=False)





def get_memory_context(num_lines: int = 5) -> str:
    """
    Returns a layered context to prevent semantic drift:
    1. Persistent Facts (User Profile)
    2. High-level History (The Summary)
    3. Recent Dialogue (Sliding Window of exact text)
    """
    global MEMORY
    context_parts = []

    # 1. LAYER ONE: PERSISTENT FACTS
    # These are core details about you that should never be misinterpreted as current requests.
    facts = MEMORY.get("persistent_facts", [])
    if facts:
        context_parts.append(f"### MASTER PROFILE (FACTS):\n" + " | ".join(facts))
    
    preferences = MEMORY.get("preferences", [])
    if preferences:
        context_parts.append(f"### USER PREFERENCES:\n" + " | ".join(preferences))



    # 2. LAYER TWO: RECENT HISTORY SUMMARY
    # This provides the 'gist' of older parts of the conversation.
    rolling_summary = MEMORY.get("temp_rolling_summary")
    if rolling_summary:
        context_parts.append(f"### PREVIOUS CONTEXT SUMMARY:\n{rolling_summary}")

    # 3. LAYER THREE: IMMEDIATE DIALOGUE (Sliding Window)
    # We increase this to 5 lines by default so the LLM remembers the exact last few things said.
    conversations = MEMORY.get("conversations", [])[-num_lines:]
    if conversations:
        dialogue_script = []
        for i,entry in enumerate(conversations):
            u = entry.get('user_text', '')
            
            t = entry.get('timestamp', '')
             # Get the concise summary of this exchange
            if i == len(conversations)-1:
                m = entry.get('maid_reply', '')
            else:
                m = entry.get('exchange_summary', '') or entry.get('maid_reply', '')
            if u or m:
                dialogue_script.append(f"User: {u}\nMaid-Chan: {m}\nAt {t}")
        
        if dialogue_script:
            context_parts.append("### RECENT EXCHANGES:\n" + "\n".join(dialogue_script))

    # Combine with clear delimiters so the LLM knows what is 'background' vs 'immediate'.
    return "\n\n".join(context_parts)


def summarize_memory_background():
    """
    Summarizes recent conversation into a 'Conversation State'.
    This forces the model to acknowledge completed tasks.
    """
    global MEMORY
    conversations = MEMORY.get("conversations", [])
    
    if len(conversations) < 3: # Summarize sooner to keep context clean
        return

    # Take the last 8 entries for a focused look
    recent_text = ""
    for entry in conversations[-8:]:
        u = entry.get('user_text', '')
        m = entry.get('maid_reply', '')
        if u and m:
            recent_text += f"User: {u}\nMaid-Chan: {m}\n"

    # STRONGER PROMPT: Focuses on State, not narrative
    summary_prompt = f"""
    [SYSTEM: Update the Conversation State]
    Analyze the history and provide a 1-sentence state update.
    
    Rules:
    1. If a search result was already given, mark it as 'COMPLETED'.
    2. DO NOT say 'no information found' if the dialogue shows Maid-Chan already answered.
    3. If the user is just saying hello, the state is 'Casual Chat'.
    4. Answer in format {{"activity":, "consuming":, "environment":,}}. For example: {{"activity": "searching for information about X", "consuming": "tea", "environment": "cleaned, quiet room"}}
    5. If user is not eating anything, consuming = None
    History:
    {recent_text}

    Current State:"""
    
    """ activity
        consuming
        enviroment
    """
    try:
        resp = ollama.generate(model=LOGIC_MODEL, prompt=summary_prompt, options={"temperature": 0},format="json")
        # Update the rolling summary with the new literal state
        MEMORY["temp_rolling_summary"] = resp['response'].strip()
        if DEBUG: debug(f"State Update: {MEMORY['temp_rolling_summary']}")
    except Exception as e:
        if DEBUG: debug(f"Background summary failed: {e}")

def exchange_summariser(maid_reply: str, user_input: str = "") -> str:
    """
    Summarizes the last exchange into a concise format for the memory.
    This is used to create a more structured record of interactions.
    """
    global MEMORY
    if DEBUG and DETAILED_MODE: 
        debug(f"Running Exchange Summarizer...")
        info(f"User Input: {user_input}")
        info(f"Maid Reply: {maid_reply}")

    

    

    summary_prompt = f"""
[SYSTEM: MEMORY COMPRESSION]
Summarize the current interaction into a single concise "Event Fact."

RULES:
1. ACTION & EMOTION: Identify WHAT she did and her MOOD (e.g., "Scolded Master," "Helped with code," "Teased Master").
2. NO DIALOGUE: Do not use quotes. Use third-person description.
4. Only output the summary of what maid chan said/did in response to the user input.Do not include what user said in the summary.
5. Keep summary clear and simple.  

User Input: {user_input}
Maid-Chan's Full Response: {maid_reply}

Compressed Memory Output:
"""

    try:
        resp = ollama.generate(model=LOGIC_MODEL, prompt=summary_prompt, options={"temperature": 0})
        summary = resp['response'].strip()
        # Update the last conversation entry with this summary
        
         # Persist the updated summary
        if DEBUG: debug(f"Exchange Summary: {summary}")
        return summary
    except Exception as e:
        if DEBUG: debug(f"Exchange summarization failed: {e}")

#--------------------------
#vector memory
#--------------------------






#--------------------------
#ledger handling
#--------------------------
ledger=[]
preferencewords = ["like", "dislike", "prefer", "love","loves", "hate", "enjoy", "desire", "want"]  

def plan_ledger(user_input):
    today = datetime.datetime.now()


    extract_prompt = f"""
SYSTEM: You are an assistant that extracts plans from user input.

TASK: Identify any plans expressed in the user's text. For each plan, extract:

1. "plan" – the full sentence or phrase describing the plan, including any time references.
2. "activity" – the main action the user intends to perform (verb + object), e.g., "go to sleep", "call John", "watch a movie".
3. "time" – the time associated with the plan, including relative expressions like "tomorrow", "in 30 minutes", or absolute times like "March 15 at 6 PM".

RULES:
1. A plan is any statement indicating an intention to do something in the future.
2. Always extract the **activity** explicitly. If the activity is unclear, infer it reasonably from context.
3. Include **relative or absolute time expressions** in the "time" field. For example, "tomorrow at 2 PM" must be kept exactly, not shortened to just "2 PM".
4. The "plan" field should preserve the full sentence/phrase describing the plan.
5. If no time is specified, use "unspecified time".
6. Ignore all general facts, observations, preferences, codes, or unrelated text; only output plans.
7. If multiple plans exist in one sentence, split them into separate entries.
8. If a relative time like 'after dinner' or 'soon' is mentioned, put that exact phrase in the 'time' field so it isn't lost
9. If the input contains no clear plans, return an empty list [] ONLY.

INPUT:
"{user_input}"

OUTPUT FORMAT:
[
    {{  "type": "plan",
        "plan": "<full plan description>",
        "time": "<time expression or 'unspecified time'>",
        "activity": "<main action verb + object or 'unspecified activity'>"
    }}
]
"""

    resp = ollama.generate(
        model=LOGIC_MODEL,
        prompt=extract_prompt,
        options={"temperature": 0}
    )

    raw_resp = resp.response.strip()
    if DEBUG: debug(f"Plan ledger:{raw_resp}")

    try:
        plans = json.loads(raw_resp)

        for plan in plans:
            time_text = plan["time"]

            if time_text != "unspecified time":
                parsed_time = cal.parseDT(time_text, today)[0]
                plan["time"] = parsed_time.strftime("%Y/%m/%d %H:%M")

        return plans

    except Exception:
        console.print(f"[red]Failed to parse plan extraction response as JSON. Raw response:[/red] {raw_resp}")
        return None
    
def temporary_data_ledger(user_input):
    # Get current year/month/day to help the model calculate "tomorrow"
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    
    # 1. THE RIGID PROMPT
    extract_prompt = f"""
    SYSTEM: Extract any temporary data from the user input. Temporary data includes any information that is likely to change or be relevant only for a short period of time.

    RULES:
    1. Temporary data can include things like specific internet address(127.0.0.1), numbers or codes, and any information that is likely to become irrelevant soon.
    2. Ignore any general facts, observations, plans, temporary states, or personal preferences; only output temporary data.
    3. If the input does not contain a specific code, number, or password, output an empty list [] instead of trying to describe the sentence.
    4. If the data is a time or a date, it is NOT temporary data. ONLY extract alphanumeric strings that look like passwords or codes.
    5. STRICTLY FORBIDDEN: Names, projects, general statements.
    

    "{user_input}"

    OUTPUT FORMAT:
    [
        {{  "type": "temporary data",
            "Data type": "Description of the type of temporary data (e.g. 'code', 'passcode',etc)",
            "value": "The actual temporary data extracted from the input"
        }},
        ...
    ]
"""
    resp = ollama.generate(
        model=LOGIC_MODEL, 
        prompt=extract_prompt, 
        options={"temperature": 0}
    )

    raw_resp = resp.response.strip()
    if DEBUG:debug(f"Temp ledger:{raw_resp}")
    
    

    try:
        # Try to parse as JSON first
        temp_data = json.loads(raw_resp)
        return temp_data

    except Exception:
        return None
    
def preference_ledger(user_input):
    # Get current year/month/day to help the model calculate "tomorrow"
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    
    # 1. THE RIGID PROMPT
    extract_prompt = f"""
    SYSTEM: Extract any personal preferences from the user input. Personal preferences include any statements that express likes, dislikes, desires, or choices.

    RULES:
    1. Personal preferences can include things like food preferences, activity preferences, or any expressed likes/dislikes.
    2. Ignore any general facts, observations, plans, or temporary data; only output personal preferences.
    3. If the input does not contain a clear expression of preference, output an empty list [] ONLY.

    INPUT:
    "{user_input}"

    OUTPUT FORMAT:
    [
        {{  "type": "preference",
            "preference": "Description of the preference (e.g., 'likes', 'dislikes', 'desires')",
            "value": "The actual preference extracted from the input"
        }},
        ...
    ]
"""
    if DEBUG and DETAILED_MODE: debug(f"This is preferencce_ledger, calling LLM")

    resp = ollama.generate(
        model=LOGIC_MODEL, 
        prompt=extract_prompt, 
        options={"temperature": 0}
    )

    raw_resp = resp.response.strip()
    if DEBUG: debug(f"Preference ledger:{raw_resp}")
    
    

    try:
        # Try to parse as JSON first
        preferences = json.loads(raw_resp)
        return preferences

    except Exception:
        return None
    



def update_maid_ledger(user_input):
    # Get current year/month/day to help the model calculate "tomorrow"
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    global preferencewords
    
    # 1. THE RIGID PROMPT
    # Updated to handle split 'time' and 'subject' for plans
    extract_prompt = f"""
    SYSTEM: You are a strict intent classifier for a personal assistant.

    RULES:
    1. PRIORITIZE PLANS: If there is a time (tomorrow, 5pm, after break) or a specific event (birthday), the primary label is "plan".
    2. BE RESTRICTIVE WITH TEMP DATA: Only use "temporary data" for strings of numbers, passcodes, or PINs.
    3. PREFERENCES ARE GENERAL: Only use "preference" for general likes/dislikes/desires/prefer x over y/love (e.g., "I like coding", "I prefer matcha over coffee"). If the user is doing it at a specific time, it's a plan, NOT a preference.
    4. MULTI-LABEL: If the user provides a passcode AND a plan in one sentence, you MUST output BOTH labels in a list like ['plan', 'temporary data'] or ['paln', 'temporary data', 'preferences'].
    5. ignore special dates(friend's birthday) unless it is a plan.
    6. Return an empty list [] if there are no plans, temporary data, or preferences.

    
    INPUT: "{user_input}"
    OUTPUT (ONLY a Python-style list):"""

    resp = ollama.generate(
        model=LOGIC_MODEL, 
        prompt=extract_prompt, 
        options={"temperature": 0}
    )

    raw_resp = resp.response.strip()
    
    

    try:
        # Try to parse as JSON first
        
            return raw_resp

    except Exception:
        return None
    
def noise_cancler(user_input):

    clean_prompt = f"""
    SYSTEM: You are a data-density optimizer. 
    TASK: Strip social noise but keep the FACTUAL relationship between the User and the Data.
    
    RULES:
    1. USE NEUTRAL VERBS: Use "wants", "dislikes", "plans to", "prefers", or "stated". Avoid "loves" unless the user said it.
    2. PRESERVE TENSE: If a user said "I used to", use "Past Preference:". 
    3. NO FILLER: Remove "hey", "wow", "oh great", and AI-directed questions.
    4. RECONSTRUCT: For short phrases like "Gym at 9", expand to "User plans to go to the gym at 9pm".
    5. Remove casual greetings, interjections, and any text that does not contribute to a clear fact or plan. Focus on extracting the core information about the user's intentions, preferences, or plans. 
    6. DON'T ANALYSIS THE INPUT, ONLY EXTRACT THE CORE INFORMATION.
    INPUT: "{user_input}"
    CLEANED SIGNAL:"""

    resp = ollama.generate(
        model=LOGIC_MODEL, 
        prompt=clean_prompt, 
        options={"temperature": 0}
    )

    raw_resp = resp.response.strip()
    
    

    try:
        return raw_resp

    except Exception:
        return None

def organise_ledger():
    global ledger, MEMORY
    active_ledger = []

    for item in ledger:
        # Since ledger now contains dictionaries, we can check keys directly
        if not isinstance(item, dict):
            continue
        if item.get("type") == "plan":
            time_str = item.get("time", "unspecified time")
            if time_str != "unspecified time":
                try:
                    dt = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
                    if dt > datetime.datetime.now():
                        active_ledger.append(item)
                except ValueError:
                    continue
        
        elif item.get("type") == "preference":
            # Add to the global MEMORY object instead of opening a new file
            if "preferences" not in MEMORY:
                MEMORY["preferences"] = []
            if item not in MEMORY["preferences"]:
                MEMORY["preferences"].append(item)

    ledger = active_ledger
    save_memory()

def fact_extractor(user_input="", maid_reply="",user_name="User"):
    prompt = f"""
TASK: Extract atomic facts from the conversation. 

### CATEGORY SELECTION (Strict):
1. **Person Info**: Any fact about a person's life, feelings, location, or actions (e.g., jobs, moves, hobbies, beliefs). 
2. **Aviation**: Technical data about aircraft, missiles, or the aerospace industry.
3. **General Facts**: Locations, weather, or general trivia.

### RULES:
- If a fact involves a person (Sean, Owen, Chloe, Sarah), use "Person Info".
- Map "I", "me", "my" or callsigns to "{user_name}".
- "You" refers to Maid-Chan and should not be included in facts about the user. 
- Keep facts under 10 words.

### EXAMPLES:
- "I'm Captain Miller, I'm a pilot" -> {{"fact": "{user_name} is a pilot", "category": "Person Info"}}
- "Owen is gay" -> {{"fact": "Owen is gay", "category": "Person Info"}}
- "Chloe got a job at Boeing" -> {{"fact": "Chloe got a job at Boeing", "category": "Person Info"}}
- "The F-16 is a fighter" -> {{"fact": "The F-16 is a multirole fighter", "category": "Aviation"}}

user_input: "{user_input}"
maid_reply: "{maid_reply}"

Return ONLY JSON:
{{
    "facts": [
        {{"fact": "string", "category": "string"}}
    ]
}}
"""
    resp=ollama.generate(model=LOGIC_MODEL, prompt=prompt, format="json", options={"temperature": 0})
    #if DEBUG:print(f"Raw response: {resp['response']}")  # Debugging line to see the raw output
    try:
        data = json.loads(resp['response'])
        return data.get("facts", [])
    except Exception as e:
        if DEBUG: debug(f"Fact extraction failed: {e}")
        return []
             














# -------------------------
# Memory handling
# -------------------------
def handle_memory(user_input: str) -> str | None:
    """
    Returns memory context relevant to the user's question, if any.
    Now reads context from the unified 'conversations' list.
    """
    text = user_input.lower()
    snippets = []

    # Since 'facts' was a separate dict and is often merged or replaced, 
    # we focus on retrieving specific history.
    
    # Check last conversation only if user asks
    if "last time" in text or "last chat" in text or "previous conversation" in text:
        conversations = MEMORY.get("conversations", [])
        if conversations:
            # Get the last unified entry
            last_convo = conversations[-1]
            
            timestamp = last_convo.get("timestamp", "an unknown time")
            user_text = last_convo.get("user_text", "no user input remembered")
            maid_reply = last_convo.get("maid_reply", "")
            
            # Construct the snippet
            snippet = f"Our last interaction was at {timestamp}. "
            snippet += f"The user said: '{user_text}'"
            if maid_reply:
                 snippet += f" and I replied: '{maid_reply}'"
            snippet += "."
            
            snippets.append(snippet)
            
    # NOTE: If you need to search for older, related topics using the 'related_to' IDs, 
    # that more complex retrieval logic would go here.
            
    if snippets:
        return " ".join(snippets)

    return None



def migrate_memory_file():
    """Converts old memory formats to the new, unified structure."""
    print(f"Attempting to migrate {MEMORY_FILE}...")
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Memory file not found or corrupted. Migration skipped.")
        return

    new_conversations = []
    
    # 1. Process existing conversations list
    for entry in data.get("conversations", []):
        new_entry = {}
        
        # Ensure mandatory fields are present
        new_entry["id"] = entry.get("id") or str(uuid.uuid4())
        new_entry["timestamp"] = entry.get("timestamp") or datetime.datetime.now().isoformat()
        new_entry["related_to"] = entry.get("related_to") or None

        # Handle different input/reply keys (Style A, Style B, and New)
        if 'user_input' in entry and 'maid_reply' in entry:
            # Already in the new format (or partial)
            new_entry["user_input"] = entry['user_input']
            new_entry["maid_reply"] = entry['maid_reply']
        elif 'user_text' in entry and 'maid_reply' in entry:
            # Legacy Style B from your snippet
            new_entry["user_input"] = entry['user_text'] # Rename key
            new_entry["maid_reply"] = entry['maid_reply']
        elif 'text' in entry:
            # Legacy Style A (combined text): Try to extract them
            try:
                # Simple split based on your snippet format: "User: ... | Maid-Chan: ..."
                user_part, maid_part = entry['text'].split(" | Maid-Chan: ", 1)
                new_entry["user_input"] = user_part.replace("User: ", "").strip()
                new_entry["maid_reply"] = maid_part.strip()
            except ValueError:
                print(f"Warning: Could not parse combined text entry: {entry['text'][:50]}... Skipping.")
                continue
        else:
            print(f"Warning: Entry missing expected keys. Skipping: {entry}")
            continue

        new_conversations.append(new_entry)

    # 2. Create the final, unified data structure
    final_data = {
        "conversations": new_conversations,
        "last_exit_time": data.get("last_exit_time", None),
        "topics": data.get("topics", {})
    }

    # 3. Save the migrated data
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Migration successful! {len(new_conversations)} entries updated.")
    
# To run this, you would call: migrate_memory_file()
def get_actual_device():
    try:
        # Check the status of running models
        status = ollama.ps()
        
        if not status.models:
            return "No model loaded (Ollama is idle)"

        # Check the first active model
        model = status.models[0]
        
        # Logic: If size_vram matches total size, it's 100% GPU
        if model.size_vram >= model.size:
            return "GPU (RTX 3050)"
        elif model.size_vram > 0:
            return f"Hybrid (Partial GPU: {model.size_vram / model.size:.1%})"
        else:
            return "CPU"
            
    except Exception as e:
        return f"Error connecting to Ollama: {e}"


ensure_nltk_data()
time_since = get_time_since_last_exit()
init_wiki_summarizer()

#--------------------------
#Test Logs
#--------------------------
logs_path=r"./logs/Testlogs"
def testloger(logname:str, content:str):
    os.makedirs(logs_path, exist_ok=True)
    with open(os.path.join(logs_path, logname), "a", encoding="utf-8") as f:
        f.write(content + "\n\n")
    
logname=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_testlog.txt"


# Initialize global stats with neutral defaults
current_stats = {"pleasure": 0, "energy": 0, "confidence": 0, "mood": "neutral"}

def mode_switcher():
    global CHAT_MODEL, MEMORY_FILE, maidchan_mode, current_stats
    
    if maidchan_mode == "tsundere":
        CHAT_MODEL = "maid-chan-tsundere:latest"
        MEMORY_FILE = "memory/tsundere_memory.json"
        vector_memory=os.path.join(BASE_DIR, "VectorMemory", "maid_tsundere_db")
        maid_brain.update_path(vector_memory)
        # Tsundere: Low pleasure (initially), High energy, High confidence
        current_stats = {"pleasure": -3, "energy": 6, "confidence": 8, "mood": ""}
        
    elif maidchan_mode == "evil":
        CHAT_MODEL = "maid-chan-evil:latest"
        MEMORY_FILE = "memory/evil_memory.json"
        vector_memory=os.path.join(BASE_DIR, "VectorMemory", "maid_evil_db")
        maid_brain.update_path(vector_memory)
        # Evil: Very low pleasure, Mid energy, Max confidence
        current_stats = {"pleasure": -8, "energy": 4, "confidence": 10, "mood": ""}
        
    elif maidchan_mode == "streamer":
        CHAT_MODEL = "maid-chan-streamer:latest"
        MEMORY_FILE = "memory/streamer_memory.json"
        vector_memory=os.path.join(BASE_DIR, "VectorMemory", "maid_streamer_db")
        maid_brain.update_path(vector_memory)
        # Streamer: High pleasure, Max energy, Mid confidence
        current_stats = {"pleasure": 5, "energy": 8, "confidence": 4, "mood": ""}
        
    elif maidchan_mode == "normal":
        CHAT_MODEL = "maid-chan-normal:latest"
        MEMORY_FILE = "memory/normal_memory.json"
        vector_memory=os.path.join(BASE_DIR, "VectorMemory", "maid_normal_db")
        maid_brain.update_path(vector_memory)
        # Normal: Mid pleasure, Mid energy, Mid confidence
        current_stats = {"pleasure": 3, "energy": 4, "confidence": 5, "mood": ""}
        
    else:
        debug(info(f"Unknown mode: {maidchan_mode}. Falling back to default."))
        CHAT_MODEL = "maid-chan-normal:latest"
        MEMORY_FILE = "memory/memory.json"
        vector_memory=os.path.join(BASE_DIR, "VectorMemory", "maid_normal_db")
        maid_brain.update_path(vector_memory)
        current_stats = {"pleasure": 0, "energy": 0, "confidence": 0, "mood": ""}

    load_memory() # Load the default memory

def seperation_line():
    size=shutil.get_terminal_size(fallback=(80, 20))
    width = size.columns
    line = "-" * width
    console.print(f"[yellow1]{line}[/yellow1]\n\n")


# -------------------------
# Main loop
# -------------------------

def main_loop():
    global DEBUG, ENABLE_TTS, MEMORY, time_since, CHAT_MODEL, maidchan_mode, logname, MEMORY_FILE, current_stats, switch_mode, ledger, searchfunc, maid_brain, DETAILED_MODE

      # Clear the rolling summary at the start of each session
    mode_switcher() 
    load_memory()
    ensure_nltk_data() # Ensure this is called once at start
    time_since = get_time_since_last_exit()
    user_asked=""
    number_of_exchanges = 0
    MEMORY["temp_rolling_summary"] = ""
    switch_mode=False
     # Set initial mode and load corresponding memory

    


    # Welcome message logic
    user_name = User # Default
    conversations = MEMORY.get("conversations", [])
    if DEBUG:
        console.print(f"""[red]Debug[/red]The variables are:\n
                ENABLE_TTS:     {ENABLE_TTS}\n
                Stream:         {STREAM}\n
                Device:         {get_device_name()}\n
                Audio Stream:   {Audio_stream}\n
                searchfunc:     {searchfunc}\n
                Maidchan Mode:  [red1]{maidchan_mode}[/red1]\n
                CHAT_MODEL:     [orange1]{CHAT_MODEL}[/orange1]\n
                MEMORY_FILE:    [yellow1]{MEMORY_FILE}[/yellow1]\n
                current_stats:   [green]{current_stats}[/green]\n
                DETAILED_MODE:   {DETAILED_MODE}

        

                

                   
                    
""")
        

    

    console.print(f"[pale_violet_red1]Maid-Chan:[/pale_violet_red1] I'm ready! {time_since}")
    if user_name=="":
        user_name=input("What's your name, this will be how maid chan going to address you in the conversations.")

    while True:
        if switch_mode:
            mode_switcher()
            switch_mode=False  # Check if we need to switch modes and load corresponding memory


        try:
            user_input = console.input("[cyan]Type here:[/cyan] \n").strip()
            testloger(logname, f"User Input: {user_input}")
            if not user_input: continue
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "exit":
            MEMORY["last_exit_time"] = datetime.datetime.now().isoformat()
            save_memory()
            organise_ledger()
           
            console.print("[pale_violet_red1]Maid-Chan:[/pale_violet_red1] *bows* Until next time! 💕")
            break
        if user_input.lower() == "debug on":
            DEBUG = True
            maid_brain = MaidMemoryTest(debug=DEBUG)
            
            if DEBUG:
                console.print(f"Ollama is currently using: {get_actual_device()}")
                console.print("[green]Debug mode enabled.[/green]")
            continue
        if user_input.lower() == "debug off":
            DEBUG = False
            maid_brain = MaidMemoryTest(debug=DEBUG)
            if DEBUG:
                console.print(f"Ollama is currently using: {get_actual_device()}")
                console.print("[green]Debug mode disabled.[/green]")
            continue
        if user_input.lower() == "tts on":
            ENABLE_TTS = True
            if DEBUG:
                console.print("[green]TTS enabled.[/green]")
            continue
        if user_input.lower() == "tts off":
            ENABLE_TTS = False
            if DEBUG:
                console.print("[green]TTS disabled.[/green]")
            continue
        if user_input.lower() == "search on":
            searchfunc = True
            if DEBUG:
                console.print("[green]Search enabled.[/green]")
            continue
        if user_input.lower() == "search off":
            searchfunc = False
            if DEBUG:
                console.print("[green]Search disabled.[/green]")
            continue
        if user_input.lower() == "detailed mode on":
            DETAILED_MODE = True
            if DEBUG:
                console.print("[green]Detailed mode enabled.[/green]")
            continue
        if user_input.lower() == "detailed mode off":
            DETAILED_MODE = False
            if DEBUG:
                console.print("[green]Detailed mode disabled.[/green]")
            continue

        cleaned_input = re.sub(r'[^a-zA-Z0-9]', '', user_input).lower()

        if cleaned_input == "tsunderemode":
            CHAT_MODEL = "maid-chan-tsundere:latest"
            maidchan_mode = "tsundere"
            
            if DEBUG:
                info(f"maidchan mode = {maidchan_mode}")
                info(f"CHAT_MODEL = {CHAT_MODEL}")
                MEMORY.clear()
                switch_mode=True
            continue
        
        if cleaned_input == "evilmode":
            CHAT_MODEL = "maid-chan-evil:latest"
            maidchan_mode = "evil"
            if DEBUG:
                info(f"maidchan mode = {maidchan_mode}")
                info(f"CHAT_MODEL = {CHAT_MODEL}")
                switch_mode=True
                MEMORY.clear()
            continue

        if cleaned_input == "normalmode":
            CHAT_MODEL = "maid-chan-normal:latest"
            maidchan_mode = "normal"
            if DEBUG:
                info(f"maidchan mode = {maidchan_mode}")
                info(f"CHAT_MODEL = {CHAT_MODEL}")
                MEMORY.clear()
                switch_mode=True
            continue

        if cleaned_input == "streamermode":
            CHAT_MODEL = "maid-chan-streamer:latest"
            maidchan_mode = "streamer"
            if DEBUG:
                info(f"maidchan mode = {maidchan_mode}")
                info(f"CHAT_MODEL = {CHAT_MODEL}")
                MEMORY.clear()
                switch_mode=True
            continue
        
        

        cleaned_input = noise_cancler(user_input)
        if DEBUG: debug(f"Cleaned Input for Ledger: {cleaned_input}")
        intent_type = update_maid_ledger(cleaned_input)
        if DEBUG: debug(f"Identified intent types: {intent_type}")


        actual_list = ast.literal_eval(update_maid_ledger(intent_type)) 
        
        try:
            for r in actual_list:
                resultparts = r.strip().lower()
                label = str(r).strip().lower()
                

                if re.search(r"plan",label,re.IGNORECASE):
                    resultparts = plan_ledger(user_input)
                    ledger.append(resultparts)
                elif re.search(r"temp",label,re.IGNORECASE):
                    resultparts = temporary_data_ledger(user_input)
                    ledger.append(resultparts)
                elif re.search(r"preference",label,re.IGNORECASE):
                    if DEBUG and DETAILED_MODE: debug("calling prerence ledger")
                    resultparts = preference_ledger(user_input)
                    ledger.append(resultparts)
                else:
                    if DEBUG: debug(f"Unrecognized output from model: {resultparts}")
                
                
        except Exception as e:
            if DEBUG: debug(f"Error processing model output: {e}")
        

        # --- 1. LIGHTNING CLASSIFICATION & EXECUTION ---
        # We run this FIRST to see if we need to perform an action
        intent = classify_input(user_input)
        execution_status = ""
        
        if "OPEN:" in intent or "CLOSE:" in intent:
            typ, prog = intent.split(":", 1)
            prog = prog.strip().lower()
            
            if typ == "OPEN":
                # We update the status so the Chat Model knows it happened
                success = open_program(prog)
                if success:
                    execution_status = f"(System Note: You have successfully opened {prog} for the user.)"
                else:
                    execution_status = f"(System Note: You tried to open {prog} but it failed or the app is unknown.)"
            
            elif typ == "CLOSE":
                close_program(prog)
                execution_status = f"(System Note: You have closed {prog}.)"

        # --- 2. CONVERSATION LOGIC (The "Maid" reacts to the "Brain") ---
        
        # Combine user input with the secret feedback note
        # Llama 3.2 will see this and naturally say "I've opened that for you!"
        final_chat_input = user_input
        if execution_status:
            final_chat_input += f" {execution_status}"


        search_keywords = ["search", "look up", "google", "find out", "look for", "tell me about", "what is", "who is"]
        user_wants_search = any(word in user_input.lower() for word in search_keywords)
        # Confidence/Search Check (Only if no command was run)
        maid_reply = ""
        update_mood_variables(current_stats, user_input, maidchan_mode)
        if searchfunc:
            if not maid_confidence_check(user_input, MEMORY):
                if DEBUG: debug("Searching...")


                user_asked = user_input
                prompt_for_permission = f"Sean-san said: '{user_input}'. Give your opinion first, then ask if I want a web search."
    

                reply=maid_chat(prompt_for_permission) 
                testloger(logname,f"Maid Chan says(search request): {reply}")

                save_single_unified_entry(user_input, maid_reply) 
                maid_mood = identify_mood(reply)
                maid_brain.save_interaction(user_input, maid_reply,maid_mood)
                extracted_facts = fact_extractor(user_input, maid_reply,"Sean")
                for item in extracted_facts:
                    if DEBUG:
                        debug(f"Extracted Fact: {item['fact']} (Category: {item['category']})")
                    maid_brain.save_memory(item['fact'], item['category'])

                if ENABLE_TTS: audio_queue.join() 
            

  
                user_response = console.input(f"[bold green]Search? (yes/no): [/bold green]").strip().lower()
                testloger(logname, f"User response to search prompt: {user_response}")

                if user_response and "yes" in user_response.lower().split():
                    maid_reply = maid_search_webV2(user_asked)
                
            
                else:

            
                    maid_reply = maid_chat(f"The user said "+user_response+". Just acknowledge it.")
                    testloger(logname, f"Maid Chan says: {maid_reply}")
            else:
                if DEBUG: debug(f"Normal chat...")
                maid_reply = maid_chat(final_chat_input)
                testloger(logname, f"Maid Chan says: {maid_reply}")

                
        else:
            if DEBUG: debug(f"Normal chat...")
            maid_reply = maid_chat(final_chat_input)
            testloger(logname, f"Maid Chan says: {maid_reply}")

        # --- 3. POST-CHAT HOUSEKEEPING ---
        exchange_summary = exchange_summariser(maid_reply, user_input)
        save_single_unified_entry(user_input, maid_reply,exchange_summary)
        def background_tasks():
            save_maid_facts(user_input, maid_reply)
            summarize_memory_background()

        threading.Thread(target=background_tasks, daemon=True).start()

        time_since = ""
        number_of_exchanges += 1
        if number_of_exchanges >=10: # Every 5 exchanges, update the time_since for better context
            if DEBUG: debug("Updating ledger and plans ...")
            organise_ledger()


        
        extracted_facts = fact_extractor(user_input, maid_reply,"Sean")
        for item in extracted_facts:
            if DEBUG:
                debug(f"Extracted Fact: {item['fact']} (Category: {item['category']})")
            maid_brain.save_memory(item['fact'], item['category'])  # Save extracted facts to the brain for vector retrieval


        maid_mood = identify_mood(maid_reply)
        maid_brain.save_interaction(user_input, maid_reply,maid_mood)
         # Save to the "brain" for vector retrieval
            

        print() # Add spacing for readability
        if ENABLE_TTS: audio_queue.join()

if __name__ == "__main__":
    main_loop()
