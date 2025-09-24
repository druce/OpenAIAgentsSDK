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
SLEEP_TIME = 10
MAX_TOKENS=8192  # for embeddings

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
    "Governance",
    "Safety And Alignment",
    "Bias And Fairness",
    "Privacy And Surveillance",
    "Inequality",
    "Job Automation",
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
    'Semiconductor Chips',
    'Neuromorphic Computing',

    "Healthcare",
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
    'Bubble',
    'Cryptocurrency',
    'Climate',
    'Energy',
    'Nuclear',
    'Scams',
    'Privacy',
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
    'Labor Market',
    'Products',
    'Opinion',
    'Review',
    'Cognitive Science',
    'Consciousness',
    'Artificial General Intelligence',
    'Singularity',
    'Manufacturing',
    'Supply chain optimization',
    'Transportation',
    'Smart grid',
    'Recommendation systems',

    # 'Nvidia',
    # 'Google',
    # 'OpenAI',
    # 'Meta',
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
    # 'Anthropic',
    # 'Cohere',
    # 'Baidu',
    # 'Big Tech',
    # 'Samsung',
    # 'Tesla',
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

    # 'ChatGPT',
    # 'Gemini',
    # 'Claude',
    # 'Copilot',

    # 'Elon Musk',
    # 'Bill Gates',
    # 'Sam Altman',
    # 'Mustafa Suleyman',
    # 'Sundar Pichai',
    # 'Yann LeCun',
    # 'Geoffrey Hinton',
    # 'Mark Zuckerberg',
    # "Demis Hassabis",
    # "Andrew Ng",
    # "Yoshua Bengio",
    # "Ilya Sutskever",
    # "Dario Amodei",
    # "Richard Socher",
    # "Sergey Brin",
    # "Larry Page",
    # "Satya Nadella",
    # "Jensen Huang",

    'China',
    'European Union',
    'UK',
    'Russia',
    'Japan',
    'India',
    'Korea',
    'Taiwan',
]
