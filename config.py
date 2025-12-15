"""Description: Constants, including configs and prompts for AInewsbot project"""
import os
import dotenv

dotenv.load_dotenv()

REQUEST_TIMEOUT = 900
SHORT_REQUEST_TIMEOUT = 60
DOMAIN_RATE_LIMIT = 5.0

DEFAULT_CONCURRENCY = 16
MAX_CRITIQUE_ITERATIONS = 2
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

NEWSAGENTDB = 'newsletter_agent.db'
LOGDB = 'newsagent_logs.db'

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
SLEEP_TIME = 5
MAX_TOKENS = 8192  # for embeddings

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

CANONICAL_TOPICS = [
    "Policy And Regulation",
    "Economics",
    "Exports And Trade",
    "Governance",
    "Safety And Alignment",
    "Bias And Fairness",
    "Privacy And Surveillance",
    "Inequality",
    "Automation",
    'Disinformation',
    'Deepfakes',
    'Sustainability',

    "Agents",
    "Coding Assistants",

    "Virtual Assistants",
    "Chatbots",
    "Robots",
    "Autonomous Vehicles",
    "Drones",
    'Virtual And Augmented Reality',

    # 'Machine learning',
    # 'Deep Learning',
    # "Neural Networks",
    # "Generative Adversarial Networks",

    'Reinforcement Learning',
    'Language Models',
    'Transformers',
    'Gen AI',
    'Retrieval Augmented Generation',
    "Computer Vision",
    'Facial Recognition',
    'Speech Recognition And Synthesis',

    'Open Source',

    'Internet of Things',
    'Quantum Computing',
    'Brain-Computer Interfaces',

    "Hardware",
    "Infrastructure",
    'Data Centers',
    'Enterprise AI',
    'Hyperscalers',
    'Adoption and Spending',
    'Semiconductor Chips',
    'GPUs',
    'AI Compute',
    'Neuromorphic Computing',

    "Healthcare",
    "Mental Health",
    "Fintech",
    "Education",
    "Entertainment",
    "Funding",
    "Venture Capital",
    "Mergers and Acquisitions",
    "Deals",
    "IPOs",
    "Ethics",
    "Legal Issues",
    "Cybersecurity",
    "AI Doom",
    'Stocks',
    'Valuation Bubble',
    'Cryptocurrency',
    'Climate',
    'Energy',
    'Nuclear',
    'Scams',
    'Intellectual Property',
    'Customer Service',
    'Military',
    'Agriculture',
    'Testing',
    'Authors And Writing',
    'Books And Publishing',
    'TV And Film And Movies',
    'Streaming',
    'Hollywood',
    'Music',
    'Art And Design',
    'Fashion',
    'Food And Drink',
    'Travel',
    'Health And Fitness',
    'Sports',
    'Gaming',
    # 'Science',
    'Politics',
    'Finance',
    'History',
    'Society And Culture',
    'Lifestyle And Travel',
    'Jobs And Careers',
    'Labor Markets And Productivity',
    'Products',
    'Opinion',
    'Review',
    'Cognitive Science',
    'Consciousness',
    'Artificial General Intelligence',
    'Singularity',
    'Manufacturing',
    'Supply Chain Optimization',
    'Transportation',
    'Smart Grid',
    'Recommendation Systems',

    'Nvidia',
    'Google',
    'OpenAI',
    'Meta',
    'xAI',
    'Perplexity',
    # 'Apple',
    # 'Microsoft',
    # 'Perplexity',
    # 'Salesforce',
    # 'Uber',
    # 'AMD',
    # 'Netflix',
    # 'Disney',
    # 'Amazon',
    # 'Cloudflare',
    'Anthropic',
    # 'Cohere',
    # 'Baidu',
    # 'Big Tech',
    # 'Samsung',
    'Tesla',
    # 'Reddit',
    # "DeepMind",
    # "Intel",
    # "Qualcomm",
    # "Oracle",
    # "SAP",
    # "Alibaba",
    # "Tencent",
    # "Hugging Face",
    # "Stability AI",
    # "Midjourney",
    # 'WhatsApp',

    'ChatGPT',
    'Gemini',
    'Claude',
    'Copilot',
    'Grok',

    'Elon Musk',
    # 'Bill Gates',
    'Sam Altman',
    'Mustafa Suleyman',
    # 'Sundar Pichai',
    # 'Yann LeCun',
    # 'Geoffrey Hinton',
    # 'Mark Zuckerberg',
    "Demis Hassabis",
    # "Andrew Ng",
    # "Yoshua Bengio",
    # "Mira Murati",
    # "Ilya Sutskever",
    # "Dario Amodei",
    # "Richard Socher",
    # "Sergey Brin",
    # "Larry Page",
    # "Satya Nadella",
    "Jensen Huang",

    'China',
    'European Union',
    'UK',
    'Russia',
    'Japan',
    'India',
    'Korea',
    'Taiwan',
]

# Langfuse tracing configuration
_LANGFUSE_CLIENT = None


def get_langfuse_client():
    """
    Get or create singleton Langfuse client with optimized settings for batch processing.

    Returns:
        Langfuse client instance or None if tracing is disabled
    """
    global _LANGFUSE_CLIENT

    tracing_enabled = os.getenv(
        'LANGFUSE_TRACING_ENABLED', 'false').lower() == 'true'

    if not tracing_enabled:
        return None

    if _LANGFUSE_CLIENT is None:
        try:
            import langfuse
            _LANGFUSE_CLIENT = langfuse.Langfuse(
                enabled=tracing_enabled,
                # Batch up to 100 events before flushing to Langfuse backend for processing
                flush_at=100,
                flush_interval=2.0,  # Flush every 2 seconds
                max_retries=3,       # Retry failed requests
                timeout=10           # Timeout for API calls (seconds)
            )
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "Langfuse tracing enabled but langfuse package not available"
            )
            return None

    return _LANGFUSE_CLIENT


def flush_langfuse_traces():
    """
    Flush any pending Langfuse traces to ensure they are sent before program exit.
    Call this at the end of the workflow or before program termination.
    """
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        try:
            _LANGFUSE_CLIENT.flush()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to flush Langfuse traces: {e}")
