import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import ollama
from rich.console import Console
from rich.live import Live
from rich.table import Table
import time
import threading

console = Console()

#----------------------
#init constants
#----------------------
max_threads = 8 
max_token = 150

class WikiModuleError(Exception):
    """Custom exception for our Wiki Module errors."""
    pass

class WikiSummarizer:
    def __init__(self, model="llama3.2", email="myemail@example.com", debug=True):
        self.model = model
        self.headers = {"User-Agent": f"MaidChanBot/1.0 ({email})"}
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.debug = debug
        self.skip_sections = [
    "References", "See also", "External links", "Further reading", 
    "Notes", "Bibliography", "Sources", "Gallery", "Citations",
    "Footnotes", "Works cited" # Added these
]
        # Added status list and lock for thread-safe live updates
        self.status_list = []
        self.status_lock = threading.Lock()

    def _generate_status_table(self):
        """Generates a grid of all current section statuses for the Live display."""
        table = Table.grid(padding=(0, 1))
        with self.status_lock:
            for status in self.status_list:
                table.add_row(status)
        return table

    def search(self, query, limit=1):
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
            if not results:
                raise WikiModuleError(f"No Wikipedia page found for '{query}'")
            if self.debug:
                console.print(f"[green]Found Wikipedia page:[/green] {results[0]['title']}")
            return results[0]["title"]
        except Exception as e:
            raise WikiModuleError(f"Search failed: {e}")

    def _worker(self, data):
        task_id, title, text = data
        idx = task_id - 1
        if not text: return (task_id, title, None)
        
        # 1. Update status to RED/Processing (replacing the initial waiting state)
        with self.status_lock:
            self.status_list[idx] = f"[bold red]●[/bold red] [white]Section {task_id:02d}:[/white] [dim]{title}[/dim] (Processing...)"

        prompt = f"Summarize section '{title}' in 2 sentences:\n\n{text}"

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_token, "temperature": 0}
            )
            
            # 2. Update status to GREEN/Complete (replacing the red processing line)
            with self.status_lock:
                self.status_list[idx] = f"[green]●[/green] [bold white]Section {task_id:02d}:[/bold white] {title} [dim](Summary Complete)[/dim]"
            
            return (task_id, title, response.message.content.strip())
        except Exception as e:
            with self.status_lock:
                self.status_list[idx] = f"[bold yellow]![/bold yellow] Section {task_id:02d}: {title} (Error: {e})"
            return (task_id, title, f"Error: {e}")

    def get_full_summary(self, query):
        overall_start = time.time()
        
        # --- PHASE 1: Wikipedia Search & Bulk Extraction ---
        wiki_start = time.time()
        page_title = self.search(query)
        
        params = {"action": "parse", "page": page_title, "prop": "text", "format": "json"}
        response = requests.get(self.api_url, params=params, headers=self.headers)
        response.raise_for_status()
        full_html = response.json()["parse"]["text"]["*"]
        soup = BeautifulSoup(full_html, "html.parser")
        content_div = soup.find(class_="mw-parser-output")

        tasks = []
        task_id = 1
        
        # Capture Lead Section
        lead_content = []
        for element in content_div.children:
            if element.name == 'h2' or (element.name == 'div' and 'mw-heading' in element.get('class', [])):
                break
            if element.name == 'p':
                txt = element.get_text().strip()
                if len(txt) > 50: lead_content.append(txt)
        
        if lead_content:
            tasks.append((task_id, "Overview", "\n".join(lead_content)))
            task_id += 1

        # Capture Sub-sections
        # Capture Sub-sections
        all_headers = content_div.find_all(['h2', 'h3', 'div'], class_=lambda x: x is None or 'mw-heading' in x)
        for item in all_headers:
            header_el = item.find(['h2', 'h3']) if item.name == 'div' else item
            if not header_el: continue
            
            title = header_el.get_text().strip().replace("[edit]", "")
            if title in self.skip_sections or not title: continue
            
            section_content = []
            for sibling in item.find_next_siblings():
                if sibling.name in ['h2', 'h3'] or (sibling.name == 'div' and 'mw-heading' in sibling.get('class', [])):
                    break
    
                # Capture paragraphs
                if sibling.name == 'p':
                    txt = sibling.get_text().strip()
                    if len(txt) > 20: section_content.append(txt)
    
                # Capture technical specifications (usually in lists or tables)
                elif sibling.name in ['ul', 'ol', 'table']:
                    txt = sibling.get_text(separator=" ").strip()
                    if len(txt) > 20: section_content.append(txt)
            
            # --- FIX: Added these lines back INSIDE the header loop ---
            if section_content:
                tasks.append((task_id, title, "\n".join(section_content)))
                task_id += 1
        
        wiki_duration = time.time() - wiki_start

        # Prepare initial status list before starting threads
        self.status_list = [f"[grey30]○ Section {i+1:02d}: {t[1]} (Waiting)[/grey30]" for i, t in enumerate(tasks)]

        # --- PHASE 2: LLM Summarization with Live Update ---
        if self.debug:
            console.print(f"\n[bold blue]Processing {len(tasks)} sections for '{page_title}'...[/bold blue]\n")
        
        llm_start = time.time()
        
        # Live context manager handles the "replacement" UI logic
        with Live(self._generate_status_table(), refresh_per_second=10) as live:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Use executor.submit to start threads and update their individual statuses
                futures = [executor.submit(self._worker, task) for task in tasks]
                
                # Keep the Live display active until all threads finish
                while any(f.running() for f in futures):
                    live.update(self._generate_status_table())
                    time.sleep(0.1)
                
                results = [f.result() for f in futures]
        
        results.sort(key=lambda x: x[0])
        llm_duration = time.time() - llm_start
        total_duration = time.time() - overall_start
        
        if self.debug:
            console.print("\n" + "─"*40)
            console.print(f"[yellow]WIKI FETCH:[/yellow] {wiki_duration:.2f}s")
            console.print(f"[blue]LLM COMPUTE:[/blue] {llm_duration:.2f}s")
            console.print(f"[bold green]TOTAL TIME:[/bold green]  {total_duration:.2f}s")
            console.print("─"*40 + "\n")
        
        return results