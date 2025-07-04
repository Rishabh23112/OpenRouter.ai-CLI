import logging
import argparse
import os
import sys
from dataclasses import dataclass
import requests
import re
from typing import List, Dict, Any, Optional
import uuid
import json
from datetime import datetime


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

#  models from OpenRouter
AVAILABLE_MODELS = [
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large",
    "mistralai/mistral-nemo",
    "google/gemini-pro-1.5",
    "anthropic/claude-3.5-sonnet-20240620",
    "cohere/command-r-plus",
    
    # can add more
]

logger = logging.getLogger(__name__)

class ApiKeyNotFoundException(Exception):
    pass


class InvalidSelectedModelException(Exception):
    pass


def display(
    message: str,
    color: str = "white",
    bold: bool = False,
    bg_color: str = "black",
):
    colors = {
        "red": "91m",
        "green": "92m",
        "yellow": "93m",
        "blue": "94m",
        "white": "97m",
    }
    bg_colors = {
        "black": "40",
        "red": "41",
        "green": "42",
        "yellow": "43",
        "blue": "44",
        "white": "47",
    }
    if bold:
        print(f"\033[1;{bg_colors[bg_color]};{colors[color]} {message}\033[0m")
    else:
        print(f"\033[{bg_colors[bg_color]};{colors[color]} {message}\033[0m")


@dataclass(frozen=True)
class ApiConfig:
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str | None = None
    usage: bool = False
    model: str | None = None


class ModelValidator:
    @staticmethod
    def validate(model: str | None) -> bool:
        if model is None:
            return False
        return model in AVAILABLE_MODELS

    @staticmethod
    def get_AVAILABLE_MODELS() -> list[str]:
        return AVAILABLE_MODELS


class ApiKeyValidator:
    @staticmethod
    def get_api_key_from_system() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")


class OpenRouter:
    def __init__(self, args) -> None:
        self.setup = ApiConfig
        # Handle None model by using default
        if args.model is None:
            args.model = "openai/gpt-4o"  # default model
        if not ModelValidator.validate(args.model):
            raise InvalidSelectedModelException(
                f"Invalid model: {args.model}\n"
                f"Available models: {ModelValidator.get_AVAILABLE_MODELS()}"
            )
        self.setup.model = args.model
        self.setup.usage = args.usage
        self.use_glow = args.glow
        if not args.api_key:
            api_key = ApiKeyValidator.get_api_key_from_system()
            if api_key is None:
                display("API key not found in environment variables! ", "red")
                logger.debug("API key not found in environment variables!")
                raise ApiKeyNotFoundException
            else:
                logger.debug(f"API key found in environment variables")
                self.setup.api_key = api_key
        else:
            self.setup.api_key = args.api_key

    def get_response(self, message) -> tuple[str, dict]:
        headers = {
            "Authorization": f"Bearer {self.setup.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  
            "X-Title": "OpenRouter CLI",  
            "Content-Type": "application/json"
        }
        logger.debug(f"Headers: {headers}")
        
        query_data = {
            "model": self.setup.model,
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": message},
            ],
            "max_tokens": 1000,  # Limit tokens for free-tier 
        }
        logger.debug(f"Query data: {query_data}")

        try:
            response = requests.post(
                self.setup.api_url, 
                headers=headers, 
                json=query_data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            if self.setup.usage:
                self._show_usage(result.get("usage", {}), self.use_glow)
            self._show_content(result["choices"][0]["message"]["content"])
            return result["choices"][0]["message"]["content"], result.get("usage", {})
        except requests.exceptions.HTTPError as err:
            if response.status_code == 401:
                display("Invalid API key! ", "red")
            else:
                display(f"HTTP Error: {err}", "red")
            logger.error(f"HTTP Error: {err}\nResponse: {response.text}")
            sys.exit(1)
        except requests.exceptions.RequestException as err:
            display(f"Request Error: {err}", "red")
            logger.error(f"Request Error: {err}")
            sys.exit(1)
        except KeyError as err:
            display(f"Unexpected response format: {err}", "red")
            logger.error(f"KeyError: {err}\nResponse: {response.text}")
            sys.exit(1)

    @staticmethod
    def _show_usage(result: dict, use_glow: bool) -> None:
        if use_glow:
            print("# Tokens")
        else:
            display("Tokens \n", "yellow", True, "blue")
        for token in result:
            print(f"- {token}: {result[token]}")
        print("\n")

    def _show_content(self, result: str) -> None:
        if self.use_glow:
            print("# Response")
        else:
            display("Response \n", "yellow", True, "blue")
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='OpenRouter CLI - AI chat with file analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openrouter.py "Hello"                           # Basic chat
  python3 openrouter.py "Analyze this" -f file.py        # File analysis
  python3 openrouter.py --live                           # Interactive mode
  python3 openrouter.py --history                        # View sessions
  python3 openrouter.py --model                          # List models
        """
    )
    parser.add_argument("query", type=str, nargs="?", help="Your question or prompt")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument("-u", "--usage", action="store_true", help="Show token usage")
    parser.add_argument("-g", "--glow", action="store_true", help="Simple output")
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        help="API key (default: OPENROUTER_API_KEY env var)",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model to use (default: openai/gpt-4o). Use -m alone to see models",
        required=False,
        default="openai/gpt-4o",
        nargs='?',  # Make it optional
    )
    # File processing
    parser.add_argument(
        "-f", "--file", nargs="*", type=str, 
        help="Files to process: PDF, TXT, MD, PY, JS, JAVA", 
        required=False
    )
    parser.add_argument(
        "--context-depth", type=int, default=3, 
        help="Number of file chunks to use (default: 3)", 
        required=False
    )
    parser.add_argument(
        "--preview", action="store_true", 
        help="Preview file chunks before sending", 
        required=False
    )
    # Session management
    parser.add_argument(
        "--live", action="store_true", 
        help="Interactive mode", 
        required=False
    )
    parser.add_argument(
        "--history", action="store_true", 
        help="List recent sessions", 
        required=False
    )
    parser.add_argument(
        "--session", type=str, 
        help="View conversation history", 
        required=False
    )
    parser.add_argument(
        "--continue", dest="continue_session", type=str, 
        help="Continue a session", 
        required=False
    )
    args = parser.parse_args()

    # Show available models if only -m is used 
    if args.model is None and ("-m" in sys.argv or "--model" in sys.argv):
        display("🤖 Available Models:", "blue", True)
        display("=" * 50, "white")
        for i, model in enumerate(AVAILABLE_MODELS, 1):
            # Color code for different models
            if "openai" in model:
                color = "green"
            elif "meta-llama" in model:
                color = "yellow"
            elif "mistralai" in model:
                color = "blue"
            elif "google" in model:
                color = "red"
            elif "anthropic" in model:
                color = "magenta"
            elif "cohere" in model:
                color = "cyan"
            else:
                color = "white"
            display(f"{i:2d}. {model}", color)
        display("=" * 50, "white")
        display("\n💡 Usage: python3 openrouter.py 'your query' -m model_name", "yellow")
        display("📝 Example: python3 openrouter.py 'Hello' -m openai/gpt-4o", "yellow")
        return

    # Ensure .embeddings/ and .sessions/ directories exist
    for d in [".embeddings", ".sessions"]:
        if not os.path.exists(d):
            os.makedirs(d)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.debug(f"args: {args}")

    # Handle session management commands first
    if args.history:
        try:
            session_manager = SessionManager()
            sessions = session_manager.list_sessions()
            if not sessions:
                display("No sessions found.", "yellow")
            else:
                display("Recent Sessions:", "blue", True)
                for session in sessions:
                    # Calculate last activity time
                    last_activity = session.get('last_activity', session.get('created_at', 'Unknown'))
                    if last_activity != 'Unknown':
                        last_activity = last_activity[:19]  # Show only date and time, not microseconds
                    
                    display(f"ID: {session['session_id'][:8]}... | Model: {session['model']} | Turns: {session['num_entries']} | Created: {session['created_at'][:19]} | Last: {last_activity}", "white")
                display("\nUse '--session <session_id>' to view full conversation history.", "yellow")
        except Exception as e:
            display(f"Error listing sessions: {e}", "red")
            sys.exit(1)
        return

    if args.session:
        try:
            session_manager = SessionManager()
            session_data = session_manager.load_session(args.session)
            display(f"Session: {args.session[:8]}... (Model: {session_data['model']})", "blue", True)
            display(f"Created: {session_data['created_at'][:19]}", "white")
            display(f"Total turns: {len(session_data['entries'])}", "white")
            display("-" * 60, "white")
            
            for i, entry in enumerate(session_data['entries'], 1):
                role = entry['role'].upper()
                content = entry['content']
                timestamp = entry['timestamp'][:19]
                
                if role == 'USER':
                    display(f"\n[{timestamp}] USER:", "green", True)
                else:
                    display(f"\n[{timestamp}] ASSISTANT:", "blue", True)
                
                # Truncate long content for display
                if len(content) > 500:
                    display(f"{content[:500]}...", "white")
                    display(f"[Content truncated. Full length: {len(content)} chars]", "yellow")
                else:
                    display(content, "white")
                
                # Show usage if available
                if 'usage' in entry:
                    try:
                        usage = json.loads(entry['usage'])
                        display(f"Tokens: {usage.get('total_tokens', 'N/A')}", "yellow")
                    except:
                        pass
        except Exception as e:
            display(f"Error loading session: {e}", "red")
            sys.exit(1)
        return

    if args.continue_session:
        try:
            session_manager = SessionManager()
            session_data = session_manager.load_session(args.continue_session)
            display(f"Continuing session {args.continue_session[:8]}... (Model: {session_data['model']})", "green")
            # TODO: Implement session continuation logic
            display("Session continuation not yet implemented.", "yellow")
        except Exception as e:
            display(f"Error loading session: {e}", "red")
            sys.exit(1)
        return

    if args.live:
        try:
            session_manager = SessionManager()
            session_id = session_manager.create_session(args.model)
            display(f"Interactive mode started. Session ID: {session_id[:8]}...", "green")
            display("Type 'quit' or 'exit' to end the session.", "yellow")
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    if not user_input:
                        continue
                    
                    # Process the input and get assistant response
                    openrouter = OpenRouter(args)
                    assistant_response, usage = openrouter.get_response(user_input)
                    
                    # Save session
                    session_manager.append_entry(session_id, "user", user_input)
                    session_manager.append_entry(session_id, "assistant", assistant_response, usage=usage)
                    
                except KeyboardInterrupt:
                    display("\nSession ended.", "yellow")
                    break
                except Exception as e:
                    display(f"Error: {e}", "red")
        except Exception as e:
            display(f"Error starting interactive mode: {e}", "red")
            sys.exit(1)
        return

    # if --usage is used without a query
    if args.usage and not args.query:
        display("📊 Usage Information:", "blue", True)
        display("=" * 40, "white")
        display("The --usage flag shows token usage information for API calls.", "white")
        display("This helps you track your token consumption and costs.", "white")
        display("\n💡 Usage Examples:", "yellow")
        display("  python3 openrouter.py 'your query' --usage", "green")
        display("  python3 openrouter.py 'your query' -u", "green")
        display("\n📈 What you'll see:", "yellow")
        display("  • Prompt tokens (input)", "white")
        display("  • Completion tokens (output)", "white")
        display("  • Total tokens used", "white")
        return

    # Check if there's a query to process
    if not args.query:
        display("❌ No query provided.", "red")
        display("💡 Use --help for usage information.", "yellow")
        display("📝 Example: python3 openrouter.py 'Hello, how are you?'", "yellow")
        return

    # File processing, embedding, and context injection
    context_chunks = []
    if args.file:
        try:
            file_processor = FileProcessor()
            embedding_manager = EmbeddingManager()
            for filepath in args.file:
                file_id = os.path.basename(filepath)
                chunks = file_processor.process_files([filepath])
                embedding_manager.ensure_embeddings(chunks, file_id)
                # Use the query or a placeholder if not present
                query = args.query or "Summarize the file."
                top_chunks = embedding_manager.search(query, file_id, top_k=args.context_depth)
                context_chunks.extend(top_chunks)
            if args.preview:
                display("File Chunk Preview:", "blue", True)
                for chunk in context_chunks:
                    display(f"[{chunk['file']}][{chunk['chunk_index']}] {chunk['content'][:200]}...", "yellow")
        except Exception as e:
            display(f"File/context error: {e}", "red")
            sys.exit(1)

    # Token limit warning (rough estimate)
    total_tokens = sum(estimate_tokens(chunk['content']) for chunk in context_chunks)
    if args.query:
        total_tokens += estimate_tokens(args.query)
    if total_tokens > 3500:
        display(f"Warning: Context + query may exceed model token limit (~4k).", "red")

    # Inject context into system prompt if present
    system_prompt = "Be precise and concise."
    if context_chunks:
        context_text = "\n\n".join(f"[File: {c['file']} | Chunk: {c['chunk_index']}]:\n{c['content']}" for c in context_chunks)
        system_prompt = f"Relevant file context:\n{context_text}\n\n{system_prompt}"

    try:
        openrouter = OpenRouter(args)
        # Patch system prompt if context is present
        if context_chunks:
            def patched_get_response(self, message):
                headers = {
                    "Authorization": f"Bearer {self.setup.api_key}",
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "OpenRouter CLI",
                    "Content-Type": "application/json"
                }
                logger.debug(f"Headers: {headers}")
                query_data = {
                    "model": self.setup.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    "max_tokens": 1000,
                }
                logger.debug(f"Query data: {query_data}")
                try:
                    response = requests.post(
                        self.setup.api_url,
                        headers=headers,
                        json=query_data,
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    if self.setup.usage:
                        self._show_usage(result.get("usage", {}), self.use_glow)
                    self._show_content(result["choices"][0]["message"]["content"])
                    return result["choices"][0]["message"]["content"], result.get("usage", {})
                except requests.exceptions.HTTPError as err:
                    if response.status_code == 401:
                        display("Invalid API key! ", "red")
                    else:
                        display(f"HTTP Error: {err}", "red")
                    logger.error(f"HTTP Error: {err}\nResponse: {response.text}")
                    sys.exit(1)
                except requests.exceptions.RequestException as err:
                    display(f"Request Error: {err}", "red")
                    logger.error(f"Request Error: {err}")
                    sys.exit(1)
                except KeyError as err:
                    display(f"Unexpected response format: {err}", "red")
                    logger.error(f"KeyError: {err}\nResponse: {response.text}")
                    sys.exit(1)
            import types
            openrouter.get_response = types.MethodType(patched_get_response, openrouter)
        
        # Save to session for all queries 
        session_manager = SessionManager()
        session_id = session_manager.create_session(args.model)
        
        # Get response and save to session
        assistant_response, usage = openrouter.get_response(args.query)
        # Save only the actual message content, not flags
        session_manager.append_entry(session_id, "user", args.query)
        session_manager.append_entry(session_id, "assistant", assistant_response, usage=usage)
        
        # Show session info if verbose or if user wants to know
        if args.verbose:
            display(f"Session saved: {session_id[:8]}...", "green")
            
    except Exception as e:
        logger.debug(f"An error occurred: {str(e)}")
        display(f"Error: {str(e)}", "red")
        sys.exit(1)


#  function for token estimation  (whitespace split for now)
def estimate_tokens(text: str) -> int:
    return len(text.split())

class FileProcessor:
    SUPPORTED_CODE_EXT = {'.py', '.js', '.java'}
    SUPPORTED_TEXT_EXT = {'.txt', '.md'}
    SUPPORTED_PDF_EXT = {'.pdf'}

    def __init__(self, chunk_min_tokens=500, chunk_max_tokens=800):
        self.chunk_min_tokens = chunk_min_tokens
        self.chunk_max_tokens = chunk_max_tokens

    def process_files(self, filepaths: List[str]) -> List[Dict[str, Any]]:
        """
        Returns a list of chunks with metadata for all files.
        """
        all_chunks = []
        for filepath in filepaths:
            ext = os.path.splitext(filepath)[1].lower()
            try:
                if ext in self.SUPPORTED_PDF_EXT:
                    chunks = self._process_pdf(filepath)
                elif ext in self.SUPPORTED_TEXT_EXT:
                    chunks = self._process_text(filepath)
                elif ext in self.SUPPORTED_CODE_EXT:
                    chunks = self._process_code(filepath, ext)
                else:
                    display(f"Unsupported file type: {filepath}", "red")
                    continue
                all_chunks.extend(chunks)
            except Exception as e:
                display(f"Failed to process {filepath}: {e}", "red")
        return all_chunks

    def _process_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        if PdfReader is None:
            raise ImportError("PyPDF2 is not installed.")
        reader = PdfReader(filepath)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return self._chunk_text(text, filepath, 'pdf')

    def _process_text(self, filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self._chunk_text(text, filepath, 'text')

    def _process_code(self, filepath: str, ext: str) -> List[Dict[str, Any]]:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        if ext == '.py':
            pattern = r'^(def |class )'
        elif ext == '.js':
            pattern = r'^(function |class |const |let |var )'
        elif ext == '.java':
            pattern = r'^(public |private |protected |class |void |int |String )'
        else:
            pattern = r'^'
        # Split on function/class definitions
        lines = code.splitlines()
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        for line in lines:
            if re.match(pattern, line.strip()) and current_tokens >= self.chunk_min_tokens:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'file': filepath,
                    'type': 'code',
                    'chunk_index': chunk_idx
                })
                chunk_idx += 1
                current_chunk = []
                current_tokens = 0
            current_chunk.append(line)
            current_tokens += estimate_tokens(line)
            if current_tokens >= self.chunk_max_tokens:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'file': filepath,
                    'type': 'code',
                    'chunk_index': chunk_idx
                })
                chunk_idx += 1
                current_chunk = []
                current_tokens = 0
        # Add any remaining lines
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'file': filepath,
                'type': 'code',
                'chunk_index': chunk_idx
            })
        return chunks

    def _chunk_text(self, text: str, filepath: str, filetype: str) -> List[Dict[str, Any]]:
        # Split on paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        for para in paragraphs:
            tokens = estimate_tokens(para)
            if current_tokens + tokens > self.chunk_max_tokens and current_tokens >= self.chunk_min_tokens:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'file': filepath,
                    'type': filetype,
                    'chunk_index': chunk_idx
                })
                chunk_idx += 1
                current_chunk = []
                current_tokens = 0
            current_chunk.append(para)
            current_tokens += tokens
        # Add any remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'file': filepath,
                'type': filetype,
                'chunk_index': chunk_idx
            })
        return chunks


class EmbeddingManager:
    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # for MiniLM-L6-v2

    def __init__(self, embedding_dir: str = ".embeddings"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        if faiss is None:
            raise ImportError("faiss-cpu is not installed.")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.embedding_dir = embedding_dir
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)

    def _get_index_path(self, file_id: str) -> str:
        return os.path.join(self.embedding_dir, f"{file_id}.index")

    def embed_chunks(self, chunks: List[dict], file_id: str) -> None:
        texts = [chunk['content'] for chunk in chunks]
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        index = faiss.IndexFlatL2(self.EMBEDDING_DIM)  # type: ignore
        index.add(vectors)  # type: ignore
        faiss.write_index(index, self._get_index_path(file_id))  # type: ignore
        # Save chunk metadata
        meta_path = os.path.join(self.embedding_dir, f"{file_id}.meta.json")
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)

    def search(self, query: str, file_id: str, top_k: int = 3) -> List[dict]:
        index_path = self._get_index_path(file_id)
        meta_path = os.path.join(self.embedding_dir, f"{file_id}.meta.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No embeddings found for {file_id}. Run embedding first.")
        index = faiss.read_index(index_path)  # type: ignore
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        query_vec = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        D, I = index.search(query_vec, top_k)  # type: ignore
        # Return top_k chunks with metadata
        return [chunks[i] for i in I[0] if i < len(chunks)]

    def ensure_embeddings(self, chunks: List[dict], file_id: str) -> None:
        index_path = self._get_index_path(file_id)
        meta_path = os.path.join(self.embedding_dir, f"{file_id}.meta.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            self.embed_chunks(chunks, file_id)


class SessionManager:
    def __init__(self, session_dir: str = ".sessions"):
        self.session_dir = session_dir
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir, f"{session_id}.json")

    def create_session(self, model: str) -> str:
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "model": model,
            "entries": []
        }
        with open(self._session_path(session_id), 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        return session_id

    def append_entry(self, session_id: str, role: str, content: str, usage: Optional[Dict[str, Any]] = None, context: Optional[List[Any]] = None):
        path = self._session_path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session {session_id} not found.")
        with open(path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content,
        }
        if usage is not None:
            entry["usage"] = json.dumps(usage)
        if context is not None:
            entry["context"] = json.dumps(context)
        session_data["entries"].append(entry)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        files = [f for f in os.listdir(self.session_dir) if f.endswith('.json')]
        sessions = []
        for fname in files:
            try:
                with open(os.path.join(self.session_dir, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get("entries", [])
                    last_activity = data.get("created_at")
                    if entries:
                        last_activity = entries[-1].get("timestamp", last_activity)
                    
                    sessions.append({
                        "session_id": data.get("session_id"),
                        "created_at": data.get("created_at"),
                        "last_activity": last_activity,
                        "model": data.get("model"),
                        "num_entries": len(entries)
                    })
            except Exception as e:
                logger.debug(f"Error reading session file {fname}: {e}")
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions[:limit]

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session {session_id} not found.")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    main()