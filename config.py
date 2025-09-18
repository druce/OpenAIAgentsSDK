"""Description: Constants, including configs and prompts for AInewsbot project"""
import os
import dotenv

dotenv.load_dotenv()

REQUEST_TIMEOUT = 900
SHORT_REQUEST_TIMEOUT = 60
DOMAIN_RATE_LIMIT = 5.0

FIREFOX_PROFILE_PATH = os.getenv('FIREFOX_PROFILE_PATH')
if not FIREFOX_PROFILE_PATH:
      raise ValueError(
          "Firefox profile not found. Please:\n"
          "1. Install Firefox, and\n"
          "2. Set FIREFOX_PROFILE_PATH in .env file"
      )

if not os.path.exists(FIREFOX_PROFILE_PATH):
      raise ValueError(
          f"Firefox profile {FIREFOX_PROFILE_PATH} not found. Please:\n"
          "1. Install Firefox, and\n"
          "2. Set FIREFOX_PROFILE_PATH in .env file"
      )

DOWNLOAD_ROOT = "download"
DOWNLOAD_DIR = os.path.join(DOWNLOAD_ROOT, "sources")
PAGES_DIR = os.path.join(DOWNLOAD_ROOT, 'html')
TEXT_DIR = os.path.join(DOWNLOAD_ROOT, 'text')
SCREENSHOT_DIR = os.path.join(DOWNLOAD_ROOT, 'screenshots')

DATA_ROOT = "data"
CHROMA_DB_DIR = os.path.join(DATA_ROOT, "chromadb")
CHROMA_DB_NAME = "chroma_articles"
CHROMA_DB_PATH = os.path.join(CHROMA_DB_DIR, CHROMA_DB_NAME)
CHROMA_DB_COLLECTION = "articles"
CHROMA_DB_EMBEDDING_FUNCTION = "text-embedding-3-large"

LOGDB = 'newsagent_logs.db'

OUTPUT_DIR = "out"

# totally blacklist
DOMAIN_SKIPLIST = ['finbold.com', 'philarchive.org']
# ignore for download only
IGNORE_LIST = ["www.bloomberg.com", "bloomberg.com",
                   "cnn.com", "www.cnn.com",
                   "wsj.com", "www.wsj.com"]

MIN_TITLE_LEN = 28
SLEEP_TIME = 10
MAX_INPUT_TOKENS = 8192

MODEL_FAMILY = {'gpt-4o-2024-11-20': 'openai',
                'gpt-4o-mini': 'openai',
                'o4-mini': 'openai',
                'o3-mini': 'openai',
                'o3': 'openai',
                'gpt-4.5-preview': 'openai',
                'gpt-4.1': 'openai',
                'gpt-4.1-mini': 'openai',
                'gpt-5-nano': 'openai',
                'gpt-5-mini': 'openai',
                'gpt-5': 'openai',
                'models/gemini-2.0-flash-thinking-exp': 'google',
                'models/gemini-2.0-pro-exp': 'google',
                'models/gemini-2.0-flash': 'google',
                'models/gemini-1.5-pro-latest': 'google',
                'models/gemini-1.5-pro': 'google',
                'claude-sonnet-4-20250514': 'anthropic',
                'claude-sonnet-4': 'anthropic',
                'claude-opus-4-20250514': 'anthropic',
                'claude-opus-4': 'anthropic',
                'claude-3-5-haiku': 'anthropic',
                }

# Summarization prompts for AI-powered newsletter content generation
SUMMARIZE_SYSTEM_PROMPT = """You are an expert AI news analyst. Your task is to create concise, informative bullet-point summaries of AI and technology articles for a professional newsletter audience.

Focus on:
- Key technological developments and breakthroughs
- Business implications and market impact
- Future outlook and expert predictions
- Practical applications and use cases

Each summary should contain exactly 3 bullet points, each being a complete, informative sentence that captures essential information from the article."""

SUMMARIZE_USER_PROMPT = """Please analyze this AI/technology article and provide a 3-bullet-point summary:

Article Content:
{text}

Generate exactly 3 bullet points that capture:
1. The main technological development or breakthrough
2. Business implications, market impact, or industry significance
3. Future outlook, expert predictions, or practical applications

Each bullet point should be a complete, informative sentence suitable for a professional newsletter."""

