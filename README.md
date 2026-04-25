# Newsletter Agent System

An AI-powered newsletter generation system built on the OpenAI Agents SDK that implements a 9-step workflow to create a news summary autonomously from multiple news sources.

## Overview

This system gathers news articles from RSS feeds, HTML scraping targets, and REST APIs; filters them for AI-related content, downloads full article text, generates summaries, clusters articles by topic, rates content quality, and produces a polished newsletter ready for distribution. Used to help generate [Skynet and Chill](https://skynetandchill.beehiiv.com/).

```mermaid
  flowchart TD
      Start([Start Newsletter Workflow]) --> Step1[1: Gather URLs<br/>RSS + HTML + REST APIs<br/>17+ sources]

      Step1 --> Step2[2: Filter URLs<br/>Deduplicate Previously Seen<br/>LLM AI Classification<br/>]

      Step2 --> Step3[3: Download Articles<br/>Playwright Scraping<br/>Content Extraction]

      Step3 --> Step4[4: LLM Summarization<br/>Bullet-point summaries]

      Step4 --> Step5[5: Rate Articles<br/>Identify<br/>on-topic, high quality<br/>using Elo-type Comparisons]

      Step5 --> Step6[6: Cluster by Topic<br/>HDBSCAN on UMAP<br/>Dimensionality-Reduced<br/>Embeddings]

      Step6 --> Step7[7: Select Sections<br/>Top Articles by Topic<br/>Newsletter Outline]

      Step7 --> Step8[8: Draft Sections<br/>Write Headlines & Content<br/>Run Critic-Optimizer Loop]

      Step8 --> Critique8[Section Critique<br/>Quality Evaluation]
      Critique8 --> Check8{Score >= 8.0?}
      Check8 -->|No & iter < 3| Improve8[Apply Improvements]
      Improve8 --> Critique8
      Check8 -->|Yes or iter = 3| Step9[9: Finalize Newsletter<br/>Generate Title<br/>Assemble Sections]

      Step9 --> Critique9[Full Newsletter<br/>Critic-Optimizer Loop<br/>Evaluate Quality<br/>5 dimension scores]

      Critique9 --> Check9{Score >= 8.0<br/>OR<br/>iter = 3?}

      Check9 -->|No| Optimizer[Optimizer Agent<br/>Apply Recommendations<br/>Improve Newsletter]
      Optimizer --> Critique9

      Check9 -->|Yes| Final([Publication-Ready<br>Newsletter<br/>Quality Score >= 8.0])

      style Step1 fill:#e1f5ff
      style Step2 fill:#e1f5ff
      style Step3 fill:#e1f5ff
      style Step4 fill:#fff9c4
      style Step5 fill:#fff9c4
      style Step6 fill:#fff9c4
      style Step7 fill:#e8f5e9
      style Step8 fill:#e8f5e9
      style Critique8 fill:#ffe0e0
      style Check8 fill:#ffe0e0
      style Improve8 fill:#ffe0e0
      style Step9 fill:#f3e5f5
      style Critique9 fill:#f3e5f5
      style Check9 fill:#f3e5f5
      style Optimizer fill:#f3e5f5
  ```

**Getting Started:**

- [`Basic OpenAI Agents SDK.ipynb`](Basic%20OpenAI%20Agents%20SDK.ipynb) - Introduction to OpenAI Agents SDK fundamentals

- [`Run Agent.ipynb`](Run%20Agent.ipynb) - Complete newsletter agent workflow with step-by-step execution

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers for web scraping
playwright install chromium

# Set up environment variables
cp dot-env.txt .env
# Edit .env with your API keys
```

### Run the Newsletter Agent

```bash
# Generate a complete newsletter
python run_agent.py

# Resume a previous session from where it left off
python run_agent.py --resume SESSION_ID

# Run a single step
python run_agent.py -s filter_urls

# Skip fetching (use cached sources)
python run_agent.py -n
```

## 📋 Complete 9-Step Workflow

The newsletter generation follows a structured 9-step process with persistent state management:

### Step 1: Gather URLs 📰

- **Sources**: Fetches from 17+ configured news sources
- **Types**: RSS feeds, HTML scraping, REST APIs
- **Output**: raw article URLs and titles

### Step 2: Filter URLs 🔍

- **AI Classification**: Uses LLM to identify AI-related content
- **Dedupe**: Filters out articles seen before in SQLite by matching URL or title+source 
- **Batch Processing**: Filter on title for AI relevance in asynchronous batches with retry
- **Output**: ~200 AI-related articles (typical 50% filter rate on 400+ new articles per day)

### Step 3: Download Articles ⬇️

- **Concurrent Scraping**: Parallel asynchronous Playwright workers with per-domain rate limiting
- **Output**:
  - Canonical URLs
  - Normalized article content (Trafilatura extraction)
  - Article metadata (publication date, open graph metadata)
  - Dedupe full text using embeddings + cosine similarity for e.g. syndicated articles on different URLs

### Step 4: Extract Summaries 📝

- **AI Summarization**: Generates bullet-point summaries for each article
- **Key Insights**: Identifies main developments and business implications
- **Output**: 3-point summaries

### Step 5: Rate Articles ⭐

- **Quality Scoring**: Assigns quality ratings (~0-10) to each article using a prompt and ELO type importance comparisons
- **Relevance**: First pass: it it spammy, on-topic, important? use LLM as binary classifier and output logrobs
- **Quality**: Run battles asking LLM to compare/rank small batches of articles according to an LLM rubric: quality, impact, novelty, etc.
- **Reputation**: articles from reputable sources get bonus points
- **Recency**: older articles get docked points
- **Frequency**: If same story is covered by several articles, pick only highest rated and boost points based on how many articles cover same set of facts
- **Output**: Deduped articles with ratings

### Step 6: Cluster by Topic 🏷️

- **Categories**: Extract free-form topics and canonical topics like AI Safety & Ethics, OpenAI, etc. via prompts; ask LLM to identify a few top topics that match articles
- **Thematic Grouping**: Organize articles into topic  using HDBSCAN over dimensionality-reduced embeddings
- **Output**: ~7 Topic tags per article and major clusters 

### Step 7: Select Sections 📑

- **Newsletter Structure**: Chooses sections, articles for each section
- **Content Planning**: Organizes articles by topic 
- **Output**: Newsletter section outline with assigned articles

### Step 8: Draft Sections ✍️

- **Content Creation**: Write engaging headlines and content for each section
- **Section Critique**: LLM evaluates section quality and provides improvement feedback
- **Iterative Refinement**: Applies critique to improve section quality, rewriting, moving or deleting articles (up to 3 iterations)
- **Output**: Polished newsletter sections with quality scores

### Step 9: Finalize Newsletter 🎉

- **Title Generation**: Creates compelling newsletter title capturing 2-3 major themes
- **Assembly**: Combines all sections into cohesive newsletter with markdown formatting
- **Critic-Optimizer Loop**: Iteratively improves newsletter quality using structured feedback
  - **Critique Agent**: Evaluates overall quality, title, structure, sections, and headlines (0-10 scores)
  - **Optimizer Agent**: Applies critique recommendations to improve newsletter
  - **Quality Threshold**: Stops when overall score ≥ 8.0 or after 3 iterations
  - **Dimension Tracking**: Monitors title_quality, structure_quality, section_quality, headline_quality
- **Output**: Publication-ready newsletter with quality score and stored title

## 🏗️ Architecture

### Core Components

#### Main Agent (`news_agent.py` + `run_agent.py`)

- `NewsletterAgent`: orchestration class implementing the 9-step workflow as OpenAI Agents SDK tools.
- `NewsletterAgentState` (in `newsletter_state.py`): Pydantic state model with SQLite persistence.
- `run_agent.py` is the CLI entry point — handles session creation, resume, single-step invocation, and graceful shutdown.

#### Source Management

- `sources.yaml`: configuration for 17+ news sources (RSS, HTML scrape, REST APIs).
- `fetch.py`: async fetching with semaphore concurrency control and per-domain rate limiting.
- `scrape.py`: Playwright-based web scraping with stealth features and infinite-scroll support.
- `utilities.py`: source validation, HTML email formatting, file output, run summaries.

#### Agent State Logic

- Top-level agent owns the state and a set of tools (one per workflow step).
- Each tool runs its step, serializes the resulting state to SQLite, and returns a status string.
- The global state has a string representation summarizing every step's results — tools can read it to make decisions.
- Prompting the agent with "run all steps in sequence" lets it iteratively inspect state and run the next logical step until done.
- If a step fails, you can fix the underlying issue and resume the same session from the failed step (`--resume SESSION_ID`) or re-run any single step (`-s STEP_NAME`).

#### RSS / HTML Source Processing

- 24-hour article filtering with robust date parsing (timezone-aware).
- Title enhancement with metadata.
- Configurable include/exclude URL regex patterns per source.
- Infinite-scroll page-down counts and post-load delays per source.

#### Concurrent Architecture

- Worker-pool-based URL processing across all sources.
- Per-domain rate limiting in scraping.
- Per-vendor async token-bucket rate limiting on LLM calls (`config.VENDOR_RPM_LIMITS`).
- Configurable concurrency from CLI (`-c N`).

#### LLM Integration (`llm.py` + `prompts.py`)

- All prompts defined in `prompts.py` as `PromptConfig` objects pinning a model and reasoning-effort level — individual steps can be retargeted by editing one line.
- **Multi-vendor support** with a unified `LLMagent` facade:
  - **OpenAI** — GPT-5 family (nano/mini/full), GPT-4.1 / GPT-4.1-mini
  - **Anthropic** — Claude Opus / Sonnet / Haiku 4.x (direct API)
  - **Google Gemini** — 3.0 Flash, 3.1 Pro, 3.1 Flash Lite
  - **OpenRouter** — Kimi K2.6, MiniMax M2.7, GLM-5.1, Grok 4.1 Fast, MiMo, Hunter Alpha, etc.
  - **Claude CLI** — invokes the local `claude -p` subprocess so the heavy editorial steps (newsletter writing, full-newsletter critique) run on your **Max-plan OAuth subscription instead of API credits**, which dramatically reduces cost
- Async token-bucket rate limiting per vendor, automatic retries with exponential backoff via `tenacity`.
- Structured output enforced by Pydantic + tool-use / JSON-schema response formats per vendor.
- **Tiered model assignment** — cheap/fast models (GPT-5-nano, Gemini Flash) for classification and extraction; mid-tier (GPT-5-mini, GPT-4.1-mini) for ratings; MiniMax / Kimi for cross-source synthesis; Claude Opus / Sonnet via CLI for the section drafting, critique, and improve loops.

#### Critic-Optimizer Loops

The pipeline uses two critic-optimizer loops to drive output quality:

- **Per-section loop** (Step 8): each section is critiqued for thematic coherence, headline quality, ordering; up to 3 iterations.
- **Whole-newsletter loop** (Step 9): full draft is scored on `title_quality`, `structure_quality`, `section_quality`, `headline_quality`, `overall_score` (0-10). Stops when overall ≥ 8.0 or after 3 iterations. Critic and optimizer run on different models (Claude Sonnet vs Claude Opus via CLI) so the optimizer doesn't just confirm its own bias.

#### Output Delivery

- Newsletter HTML is saved to `out/YYYY-MM-DD.html` (with `out/latest.html` symlinked to today's file).
- Optional Gmail SMTP delivery via `utilities.send_email`.
- Each headline in the rendered HTML includes a `⎘` copy-to-clipboard button (works in browser; Gmail strips JS).
- Email footer includes a `file://` link back to the local HTML so you can open the fully-interactive version in a browser when reading the email.

The notebook is used for:

- **Development**: Testing new features and workflow improvements
- **Debugging**: Identifying and fixing issues in the pipeline
- **Optimization**: Tuning parameters for better content quality
- **Analysis**: Understanding content patterns and classification accuracy

## 📁 Repository Structure

```bash
OpenAIAgentsSDK/
├── Basic OpenAI Agents SDK.ipynb  # Introduction to OpenAI Agents SDK
├── Run Agent.ipynb                # Newsletter workflow notebook ⭐
├── run_agent.py                   # CLI entry point ⭐
├── news_agent.py                  # NewsletterAgent orchestration class
├── newsletter_state.py            # Pydantic state model + SQLite persistence
├── prompts.py                     # All LLM prompt templates ⭐
├── sources.yaml                   # News source configurations
├── config.py                      # Constants, model registry, canonical topics
├── db.py                          # Database schema (URLs, Articles, Newsletters)
├── do_cluster.py                  # Topic clustering via HDBSCAN
├── do_dedupe.py                   # Deduplication logic
├── do_rating.py                   # Article rating with Bradley-Terry
├── fetch.py                       # Multi-source content fetching
├── scrape.py                      # Playwright-based web scraping
├── llm.py                         # Multi-vendor LLM integration (OpenAI, Anthropic, Gemini, OpenRouter, Claude CLI)
├── utilities.py                   # Email delivery, HTML formatting, source validation
├── log_handler.py                 # SQLite logging with API key sanitization
├── mcptool.py                     # MCP (Model Context Protocol) integration
├── dot-env.txt                    # Environment variables template
├── headline_classifier_ground_truth.csv  # Evaluation data
├── prompts.md                     # Prompt documentation
├── requirements.txt               # Python dependencies
├── promptfoo/                     # Prompt evaluation framework
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🔧 Configuration

### Source Configuration (`sources.yaml`)

```yaml
Source Name:
  type: rss|html|rest         # Source type
  url: "https://..."          # Source URL
  filter_24h: true           # Filter RSS to last 24 hours
  include: [...]             # URL patterns to include
  exclude: [...]             # URL patterns to exclude
  scroll: 3                  # Page scrolls for dynamic content
  initial_sleep: 5           # Load delay in seconds
```

### Environment Variables

```bash
# Required for core functionality
OPENAI_API_KEY=sk-...           # OpenAI (GPT-5 family, GPT-4.1, embeddings)
NEWSAPI_API_KEY=...             # NewsAPI for REST sources
FIREFOX_PROFILE_PATH=...        # Firefox profile for Playwright scraping

# Vendor keys (optional but referenced by various prompts in prompts.py)
ANTHROPIC_API_KEY=sk-ant-...    # Anthropic Claude (direct API)
GOOGLE_API_KEY=...              # Google Gemini
OPENROUTER_API_KEY=sk-or-...    # OpenRouter (Kimi, MiniMax, GLM, Grok)

# Claude CLI (alternative to ANTHROPIC_API_KEY for *-cli model variants)
# Uses your local `claude` CLI's OAuth (Max-plan subscription) instead of API.
# No env var needed; just have `claude` on $PATH and logged in.
```

## 🚀 Production Deployment

### Workflow Execution

```bash
# Complete workflow
python run_agent.py

# Resume an interrupted run from where it left off
python run_agent.py --resume SESSION_ID

# Re-run a single step on a previous session (e.g. regenerate the newsletter)
python run_agent.py --resume SESSION_ID -s finalize_newsletter --notify
```

### Performance Characteristics

- **Speed**: Complete workflow with 100 articles in ~15 minutes (with cached content)
- **Scalability**: Should handles 1000+ articles with concurrent processing
- **Reliability**: Automatic retry logic and error recovery
- **Quality**: High-quality AI-generated content with iterative refinement and quality scoring
- **Efficiency**: Early stopping in critic-optimizer loops when quality threshold met

### Content Quality

- **Accuracy**: LLM classification with ~95% precision on AI content
- **Relevance**: Smart filtering for high-quality, recent articles
- **Readability**: Professional newsletter formatting and structure
- **Completeness**: Full article content with proper attribution

## 📊 Typical Results

**Input**:

- 17 news sources → 650+ raw articles
- AI classification → 330+ AI-related articles
- Content download → 300+ full articles (after deduplication)

**Output**:

- Topic clusters (LLM Advances, AI Safety, Business Applications, etc.)
- 7-15 newsletter sections with 2-7 stories each
- 30+ articles in final newsletter
- 1200+ words of professional content
- Quality score: 8.0-9.5/10 (via critic-optimizer loop)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Built with ❤️ using OpenAI Agents SDK, Claude Code, Windsurf, Python, Pandas, Jupyter, Playwright and other great tools!

---
