import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import chromadb
from sentence_transformers import SentenceTransformer
import time
import hashlib
import datetime
from rich.console import Console
import os

path=os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(path)

console = Console()

# Using Nomic-embed 1.5 - High performance while staying on CPU
Model="nomic-ai/nomic-embed-text-v1.5"

class MaidMemoryTest:
    def __init__(self, debug=False, detail=False):
        self.DEBUG = debug
        self.detail = detail 
        if self.DEBUG: console.print("[gold3]--- Initializing Librarian (Memory System) ---[/gold3]\n")
        
        # CHANGED: New path to ensure a fresh start and avoid version/dimension mismatch
        self.client = chromadb.PersistentClient(path=os.path.join(parent_dir, "VectorMemory", "maid_normal_db"))
        
        # Device set to 'cpu' to preserve your RTX 3050 VRAM for the LLM
        self.model = SentenceTransformer(Model, trust_remote_code=True, device='cpu')
        
        # CHANGED: New collection name to force a clean schema
        self.collection = self.client.get_or_create_collection("MaidFinalCollection")
        if self.DEBUG: console.print("[gold3]--- Librarian Ready ---[/gold3]\n")

    def update_path(self, new_path):
        """Allows changing the database path if needed."""
        new_path = os.path.join(parent_dir, "VectorMemory", new_path)
        if self.DEBUG: print(f"Updating memory path to: {new_path}")
        self.client = chromadb.PersistentClient(path=new_path)
        self.collection = self.client.get_or_create_collection("MaidFinalCollection")
        if self.DEBUG: console.print("[gold3]Memory path updated successfully.[/gold3]\n")

    def save_memory(self, text, category="Others"):
        """Saves memory with a Hash ID to prevent exact duplicates."""
        mem_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if self.DEBUG: print(f"Processing: '{text[:50]}...'")
        embedding = self.model.encode(text, convert_to_tensor=False).tolist()
        
        self.collection.upsert(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"category": category,
                        "role": "fact",
                         "timestamp": time.time()}],
            ids=[mem_id]
        )
        
        if self.DEBUG: print(f"Memory Secured. (ID: {mem_id[:8]})\n")

    def recall(self, query):
        """Finds the most relevant memories based on meaning and includes time."""
        if self.DEBUG:
            print(f"Recalling memories related to: '{query}'...")
    
        query_vec = self.model.encode(query, convert_to_tensor=False).tolist()
    
        user_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            where={"role": "user"}
        )
        maid_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            where={"role": "maid"}
        )
        fact_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=5,
            where={"role": "fact"}
        )
        if self.DEBUG and self.detail:
            print(f"User Memories Found: {len(user_results['documents'][0]) if user_results['documents'] else 0}")
            print(f"Maid Memories Found: {len(maid_results['documents'][0]) if maid_results['documents'] else 0}")
            print(f"Fact Memories Found: {len(fact_results['documents'][0]) if fact_results['documents'] else 0}")

        
        
        answer = ""
        def parse_results(res):
            items = []
            if res['documents'] and res['documents'][0]:
                for i in range(len(res['documents'][0])):
                    items.append({
                        "doc": res['documents'][0][i],
                        "metadata": res['metadatas'][0][i],
                        "dist": res['distances'][0][i]
                    })
            return items
        
        candidates = []
        candidates.extend(parse_results(user_results))
        candidates.extend(parse_results(maid_results))
        candidates.extend(parse_results(fact_results))


        if self.DEBUG:
            print(f"Memories found:\n")
            n=0
            for entry in candidates:
                ts = entry['metadata'].get('timestamp', 0)
                readable_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                role = entry['metadata'].get('role', 'unknown')
                if n % 2 == 0:
                    console.print(f"[gold3] [{readable_time}] ({role}): {entry['doc']} [gold3](Distance: {entry['dist']:.4f})[/gold3]")
                else:
                    console.print(f"[cyan] [{readable_time}] ({role}): {entry['doc']} [cyan](Distance: {entry['dist']:.4f})[/cyan]")
                n+=1

        if not candidates:
            if self.DEBUG: print(" [!] No relevant memories found.")
            return candidates

        candidates.sort(key=lambda x: x['dist'])
        final_selection = candidates[:5]

        n=0
        num_memories_included = 0
        for entry in final_selection:
                ts = entry['metadata'].get('timestamp', 0)
                readable_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                role = entry['metadata'].get('role', 'unknown')
                threshold = 1.3  # Adjust this threshold based on your needs
        
        # We use a threshold check just like your original code
                if entry['dist'] <= threshold:
                    answer += f" [{readable_time}] ({role}): {entry['doc']}\n"
                    num_memories_included += 1
                    if self.DEBUG and self.detail:
                        if n % 2 == 0:
                            console.print(f"[gold3] Selected Memory: [{readable_time}] ({role}): {entry['doc']} [gold3](Distance: {entry['dist']:.4f})[/gold3]")
                        else:
                            console.print(f"[cyan] Selected Memory: [{readable_time}] ({role}): {entry['doc']} [cyan](Distance: {entry['dist']:.4f})[/cyan]")
                    n += 1
        if self.DEBUG:
            print(f"[bright_green]\nTotal Memories Included in Answer: {num_memories_included}[/bright_green]\n")
        return answer
        


    def clean_all_duplicates(self):
        """Force-scans the DB and removes any repeated text."""
        if self.DEBUG:
            console.print("[gold3]--- Running Deep Memory Clean ---[/gold3]\n")
        all_mems = self.collection.get()
        seen_docs = set()
        ids_to_remove = []

        for i, doc in enumerate(all_mems['documents']):
            if doc in seen_docs:
                ids_to_remove.append(all_mems['ids'][i])
            else:
                seen_docs.add(doc)

        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
            if self.DEBUG:
                console.print(f"[gold3]Cleaned {len(ids_to_remove)} duplicate entries.[/gold3]\n")
        else:
            if self.DEBUG:
                console.print("[gold3]Memory is already optimized and unique.[/gold3]\n")
                console.print("-" * 30)

    def save_interaction(self, user_text, maid_text, mood="neutral"):
        """Saves full interactions with manual embedding to fix dimension errors."""
        combined_text = f"User: {user_text}\nMaid-chan: {maid_text}"
        usermem_id = hashlib.md5(user_text.encode()).hexdigest()
        maidmem_id = hashlib.md5(maid_text.encode()).hexdigest()
        
        # FIXED: Explicitly calculating the embedding so ChromaDB doesn't guess dimensions
        user_embedding = self.model.encode(user_text, convert_to_tensor=False).tolist()
        maid_embedding = self.model.encode(maid_text, convert_to_tensor=False).tolist()
    
        self.collection.upsert(
                embeddings=[user_embedding],
                documents=[combined_text],
                metadatas=[{
                    "role": "user",
                    "user_input": user_text, 
                    "reply": maid_text,
                    "mood": mood, 
                    "timestamp": time.time()
                }],
                ids=[usermem_id]
        )

        self.collection.upsert(
                embeddings=[maid_embedding],
                documents=[combined_text],
                metadatas=[{
                    "role": "maid",
                    "user_input": user_text, 
                    "reply": maid_text,
                    "mood": mood, 
                    "timestamp": time.time()
                }],
                ids=[maidmem_id]
        )



        if self.DEBUG:   
            print(f"Interaction saved with ID: {usermem_id[:8]} (Mood: {mood})\n")

    
# --- EXECUTION TEST ---
if __name__ == "__main__":


    maid_brain = MaidMemoryTest(debug=True)
    while True:
        mode=input("Which maidchan's memory do you want to check 1. Normal 2. Tsundere 3. Evil 4. Streamer ").strip().lower()
        if mode=="1":
            maid_brain.update_path("maid_normal_db") 
            break
        elif mode=="2":
            maid_brain.update_path("maid_tsundere_db")
            break
        elif mode=="3":
            maid_brain.update_path("maid_evil_db")
            break
        elif mode=="4":
            maid_brain.update_path("maid_streamer_db")
            break
        else:
            print("Invalid choice, defaulting to Normal.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "switchmode":
            mode=input("Which maidchan's memory do you want to check 1. Normal 2. Tsundere 3. Evil 4. Streamer ").strip().lower()
            if mode=="1":
                maid_brain.update_path("maid_normal_db")
            elif mode=="2":
                maid_brain.update_path("maid_tsundere_db")
            elif mode=="3":
                maid_brain.update_path("maid_evil_db")
            elif mode=="4":
                maid_brain.update_path("maid_streamer_db")
            else:
                print("Invalid choice, defaulting to Normal.")
                maid_brain.update_path("maid_normal_db")
            continue
        res=maid_brain.recall(user_input)
        print(f"Maid-chan's Memory Recall:\n{res}")
