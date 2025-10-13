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

      Step4 --> Step5[5: Cluster by Topic<br/>HDBSCAN + Embeddings]

      Step5 --> Step6[6: Rate Articles<br/>Identify on-topic, high quality<br/>using Elo-type Comparisons]

      Step6 --> Step7[7: Select Sections<br/>Top Articles by Topic<br/>Newsletter Outline]

      Step7 --> Step8{8: Draft Sections<br/>Write Headlines & Content<br/>Critic-Optimizer Loop}

      Step8 --> Critique8[Section Critique<br/>Quality Evaluation]
      Critique8 --> Check8{Score >= 8.0?}
      Check8 -->|No & iter < 3| Improve8[Apply Improvements]
      Improve8 --> Step8
      Check8 -->|Yes or iter = 3| Step9[Step 9: Finalize Newsletter<br/>Generate Title<br/>Assemble Sections]

      Step9 --> Critique9[Critique Full Newsletter<br/>Evaluate Quality<br/>5 dimension scores]

      Critique9 --> Check9{Score >= 8.0<br/>OR<br/>iter = 3?}

      Check9 -->|No| Optimizer[Optimizer Agent<br/>Apply Recommendations<br/>Improve Newsletter]
      Optimizer --> Critique9

      Check9 -->|Yes| Final([Publication-Ready Newsletter<br/>Quality Score >= 8.0])

      style Step1 fill:#e1f5ff
      style Step2 fill:#e1f5ff
      style Step3 fill:#e1f5ff
      style Step4 fill:#e1f5ff
      style Step5 fill:#e1f5ff
      style Step6 fill:#e1f5ff
      style Step7 fill:#e8f5e9
      style Step8 fill:#e8f5e9
      style Step9 fill:#f3e5f5
      style Critique8 fill:#ffe0e0
      style Critique9 fill:#ffe0e0
      style Optimizer fill:#ffe0e0
      style Check8 fill:#fff9c4
      style Check9 fill:#fff9c4
      style Final fill:#c8e6c9
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
python news_agent.py

```

## 📋 Complete 9-Step Workflow

The newsletter generation follows a structured 9-step process with persistent state management:

### Step 1: Gather URLs 📰

- **Sources**: Fetches from 17+ configured news sources
- **Types**: RSS feeds, HTML scraping, REST APIs
- **Output**: raw article URLs and titles

### Step 2: Filter URLs 🔍

- **AI Classification**: Uses LLM to identify AI-related content
- **Dedupe**: Filters out articles seen before by matching URL or title/source (with date override to rerun)
- **Batch Processing**: Efficiently processes articles in batches
- **Output**: ~200 AI-related articles (typical 50% filter rate on 400+ new articles per day)

### Step 3: Download Articles ⬇️

- **Concurrent Scraping**: Parallel asynchronous Playwright workers with per-domain rate limiting
- **Output**:
  - Redirect URLs
  - Clean article content (trafilatura extraction)
  - Article metadata (publication date, open graph metadata)
  - Dedupe using embeddings + cosine similarity for e.g. syndicated articles on different URLs

### Step 4: Extract Summaries 📝

- **AI Summarization**: Generates bullet-point summaries for each article
- **Key Insights**: Identifies main technological developments and business implications
- **Output**: 3-point summaries stored in persistent state

### Step 5: Cluster by Topic 🏷️

- **Categories**: Apply free-form topics and canonical topics like AI Safety & Ethics, OpenAI, etc. via prompts; ask LLM to identify a few top topics that match articles
- **Thematic Grouping**: Organize articles into topic clusters using HDBSCAN over dimensionality-reduced embeddings
- **Output**: Topic tags and clusters

### Step 6: Rate Articles ⭐

- **Quality Scoring**: Assigns quality ratings (1-10) to each article using a prompt and ELO type importance comparisons
- **Relevance**: Initial pass: it it spammy, on-topic, important? use LLM and output logrobs
- **Quality**: Run battles asking LLM to compare articles according to a rubric: quality, impact, novelty, etc.
- **Reputation**: articles from reputable sources get boost points
- **Recency**: downrate old articles
- **Frequency**: If same story is covered by several articles, pick highest rated and boost points based on how many articles cover same set of facts
- **Output**: Deduped articles with ratings

### Step 7: Select Sections 📑

- **Newsletter Structure**: Chooses top articles for each section
- **Content Planning**: Organizes articles by topic clusters
- **Output**: Newsletter section outline with assigned articles

### Step 8: Draft Sections ✍️

- **Content Creation**: Writes engaging headlines and content for each section
- **Section Critique**: LLM evaluates section quality and provides improvement feedback
- **Iterative Refinement**: Applies critique to improve section quality (up to 3 iterations)
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

#### Main Agent (`news_agent.py`)

- `NewsletterAgent`: Primary orchestration class implementing the 9-step workflow
- `NewsletterAgentState`: Pydantic model for persistent state management
- Tool-based architecture using OpenAI function calling

#### Source Management

- `sources.yaml`: Configuration for 17+ news sources
- `fetch.py`: Async fetching system with concurrency control
- `scrape.py`: Advanced web scraping with Playwright and stealth features

#### State Persistence

- SQLite-based workflow state storage
- Resumable execution from any step
- Comprehensive error handling and recovery

### Key Features

#### RSS Source Processing

- 24-hour article filtering
- Title enhancement with metadata
- Robust date parsing with timezone awareness

#### HTML Source Processing

- Playwright-based browser automation
- Configurable include/exclude URL patterns
- Support for infinite scroll and dynamic content

#### Concurrent Architecture

- Worker pool-based URL processing
- Per-domain rate limiting with atomic slot acquisition
- Random URL selection to prevent domain clustering

#### AI Integration

- GPT-5-nano for headline classification
- GPT-4o for content summarization and newsletter writing
- Structured output with Pydantic validation
- Critic-optimizer loops with quality scoring and iterative refinement

## 📓 Interactive Development: test_agent.ipynb

The `test_agent.ipynb` notebook provides an interactive development environment for testing and refining the newsletter generation workflow. This notebook is essential for:

### Development Features

#### Step-by-Step Execution

- Run individual workflow steps in isolation
- Inspect intermediate results and data transformations
- Debug issues with detailed logging and output

#### Data Exploration

- Visualize article classification results
- Analyze topic clustering effectiveness
- Review content quality and summary generation

#### Configuration Testing

- Test different LLM models and prompts
- Experiment with source configurations
- Validate filtering and rating parameters

#### State Management

- Load and inspect persistent workflow state
- Reset workflow to specific steps for testing
- Export results for analysis

### Notebook Structure

1. **Environment Setup**: API keys, imports, and configuration
2. **Agent Initialization**: Create newsletter agent with persistent state
3. **Step Execution**: Run individual workflow steps with detailed output
4. **Data Analysis**: Explore results, visualize metrics, and validate quality
5. **Testing & Debugging**: Troubleshoot issues and optimize parameters

### Usage Examples

```python
# Initialize agent with persistent state
agent = NewsletterAgent(session_id="test_session", verbose=True)

# Run individual steps
await agent.gather_urls()
await agent.filter_urls()
await agent.download_articles()

# Inspect results
print(f"Articles found: {len(agent.state.headline_data)}")
ai_articles = [a for a in agent.state.headline_data if a.get('isAI')]
print(f"AI-related: {len(ai_articles)}")
```

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
├── news_agent.py                  # Main agent implementation
├── sources.yaml                   # News source configurations
├── config.py                      # System configuration
├── db.py                          # Database utilities
├── do_cluster.py                  # Topic clustering via HDBSCAN
├── do_dedupe.py                   # Deduplication logic
├── do_rating.py                   # Article rating with Bradley-Terry
├── fetch.py                       # Multi-source content fetching
├── llm.py                         # LLM integration with Langfuse
├── scrape.py                      # Web scraping utilities
├── dot-env.txt                    # Environment variables template
├── headline_classifier_ground_truth.csv  # Evaluation data
├── list_langfuse_prompts.py       # Export Langfuse prompts
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
OPENAI_API_KEY=sk-...        # OpenAI API key for LLM operations
NEWSAPI_KEY=...              # NewsAPI key for REST sources
```

## 🧪 Testing

```bash
# Run fast tests (no network requests)
pytest tests/

# Run all tests including slow network tests
pytest tests/ --run-slow

# Run with coverage reporting
pytest tests/ --cov=. --cov-report=term-missing

# Test specific functionality
pytest tests/test_sources.py::TestRSSSourceFetching -v
```

### Test Individual Sources

```bash
# Test single source
python tests/run_tests.py "Hacker News"

# Demo 24-hour filtering
python tests/demo_24h_filtering.py
```

## 🚀 Production Deployment

### Workflow Execution

```bash
# Complete workflow with monitoring
python news_agent.py

# Resume from specific step
python check_status.py
python news_agent.py  # Automatically resumes from last step
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

**Input Processing**:

- 17 news sources → 650+ raw articles
- AI classification → 330+ AI-related articles
- Content download → 300+ full articles (after deduplication)

**Output Quality**:

- 6 topic clusters (LLM Advances, AI Safety, Business Applications, etc.)
- 7-15 newsletter sections with 2-7 stories each
- 30+ articles in final newsletter
- 1200+ words of professional content
- Quality score: 8.0-9.5/10 (via critic-optimizer loop)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

### Built with ❤️ using OpenAI Agents SDK, Playwright, and advanced AI techniques**

---
