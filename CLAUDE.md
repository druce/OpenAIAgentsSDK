# CLAUDE.md

## Commands

```bash
# Run the full workflow
python run_agent.py

# Resume a previous session
python run_agent.py --resume SESSION_ID

# Run a single step
python run_agent.py -s filter_urls

# Skip fetching (use cached sources)
python run_agent.py -n

# Common options
python run_agent.py -c 12 -e 2 -t 30 -v   # concurrency, max-edits, timeout, verbose

# Tests
pytest tests/                              # fast tests (no network)
pytest tests/ --run-slow                   # include network/browser tests
pytest tests/ --cov=. --cov-report=term-missing

# Install
pip install -r requirements.txt
playwright install chromium
```

## Architecture

Newsletter agent system that fetches news from multiple sources and produces a daily AI newsletter via a 9-step workflow.

### Core Modules

| Module | Purpose |
|--------|---------|
| `run_agent.py` | CLI entry point with resume/step-based execution |
| `news_agent.py` | Main `NewsletterAgent` class orchestrating the 9-step workflow |
| `llm.py` | Multi-vendor LLM calls (Anthropic, OpenAI, Gemini, OpenRouter) with rate limiting and retries |
| `prompts.py` | LLM prompt templates |
| `config.py` | Constants, model registry (`MODEL_FAMILY`), canonical topic list |
| `fetch.py` | Async source fetching (RSS/HTML/REST) with semaphore concurrency |
| `scrape.py` | Playwright-based web scraping and content extraction |
| `newsletter_state.py` | Pydantic state model with SQLite persistence and resume support |
| `db.py` | Database schema (URLs, Articles, Newsletters, Sites) |
| `do_cluster.py` | Topic clustering via embeddings + HDBSCAN |
| `do_rating.py` | Bradley-Terry article rating |
| `do_dedupe.py` | Article deduplication and filtering |
| `utilities.py` | Email delivery, HTML formatting, source validation, run summaries |
| `log_handler.py` | SQLite logging with API key sanitization |

### Workflow Steps

1. **gather_urls** - Fetch articles from RSS, HTML, REST sources
2. **filter_urls** - AI-based content classification, dedup
3. **download_articles** - Extract full article content (trafilatura)
4. **extract_summaries** - Generate bullet-point summaries via LLM
5. **rate_articles** - Bradley-Terry comparative scoring
6. **cluster_by_topic** - HDBSCAN clustering with embeddings
7. **select_sections** - Choose top articles per topic
8. **draft_sections** - Generate section narratives
9. **finalize_newsletter** - Assemble final HTML and optional email delivery

State persists to SQLite between steps, enabling resume from any point via `--resume`.

### LLM Integration

Multi-vendor support via `llm.py`:
- **Vendors**: Anthropic, OpenAI, Google Gemini, OpenRouter
- **Rate limiting**: Async token-bucket per vendor (configurable in `config.VENDOR_RPM_LIMITS`)
- **Structured output**: Pydantic-based response parsing
- **Retry logic**: Exponential backoff via tenacity
- Model-to-vendor mapping in `config.MODEL_FAMILY`

### Source Configuration (`sources.yaml`)

```yaml
Source Name:
  type: rss|html|rest
  url: "https://..."
  rss: "https://..."           # RSS feed URL
  filter_24h: true             # Filter to last 24 hours (default: true)
  include: [...]               # URL regex include patterns
  exclude: [...]               # URL regex exclude patterns
  scroll: N                    # Infinite scroll page-downs
  initial_sleep: N             # Wait after page load (seconds)
  minlength: N                 # Minimum link text length
  function_name: "func_name"   # For REST sources
```

### Data Storage

- `newsletter_agent.db` - Agent state and workflow persistence
- `newsagent_logs.db` - Structured logs
- `download/` - Cached source HTML and extracted text
- `data/chromadb/` - Configured in `config.py` but currently unused; embeddings live in `umap_reducer.pkl` for clustering. Slated to be replaced (see in-flight KG+RAG plan).
- `out/` - Generated newsletter output
- `umap_reducer.pkl` - Pretrained UMAP reducer required by `do_cluster.py`. Not in git (414 MB). Run `Tune HDBSCAN.ipynb` to regenerate if missing.

### Environment

Requires `.env` with at minimum:
- `OPENAI_API_KEY`
- `FIREFOX_PROFILE_PATH` (for Playwright scraping)
- Vendor API keys as needed (Anthropic, Google, OpenRouter)
