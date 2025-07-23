---
title: Personalized Learning AI Recommender
emoji: 📚
colorFrom: "indigo" # Your header background color
colorTo: "blue"   # A complementary color, e.g., your link color
sdk: gradio
sdk_version: "4.38.1" # Adjust this to your gradio version from requirements.txt if different
app_file: app.py
pinned: false
---

# 📚 Personalized Learning AI Recommender

A sophisticated agentic AI application that helps users find personalized learning resources across the web. Powered by Anthropic's Claude LLM, it understands your learning goals and preferences to curate highly relevant content from various online sources.

## ✨ What makes this project special?

This project demonstrates key concepts of agentic AI and practical web development:

* **Agentic Capabilities:**
    * **Intelligent Perception:** Proactively generates optimized Google Custom Search queries and performs an **LLM-powered initial reranking** of raw search results (titles/snippets) to efficiently identify the most promising links.
    * **Deep Reasoning & Planning:** Uses Anthropic's Claude LLM for sophisticated understanding, including **categorizing resources, assessing relevance, skill level, and quality with a transparent AI reasoning explanation.** It plans a multi-step process from smart search to curated, structured output.
    * **Action & Presentation:** Executes web requests, parses HTML, and generates a rich, categorized HTML report directly within the UI.
    * **Autonomy:** Operates autonomously from user input to final recommendations.
* **Enhanced Accuracy with LLMs:** Leverages Anthropic Claude's advanced semantic understanding for more precise recommendations, now with an **intelligent pre-filtering step** that reduces noise and cost.
* **Cost Efficiency & Performance:** Implements a local SQLite caching mechanism to minimize redundant web scraping and expensive LLM API calls, further optimized by targeting detailed LLM analysis *only* on highly promising search results.
* **User-Friendly Interface:** Provides an interactive web UI using Gradio, making it easy for anyone to use.
* **Transparency & Trust:** Offers **AI-generated reasoning** for each recommendation, allowing users to understand *why* a resource was chosen, fostering trust and interpretability.
* **Robustness & Production-Ready Practices:** Includes comprehensive error handling, structured logging, secure environment variable management for API keys, and adherence to web scraping politeness.

## 🚀 How It Works

1.  **User Input:** You provide a learning goal (e.g., "Python for Data Science"), a preferred learning style (e.g., "video", "articles", "hands-on projects"), and your current skill level.
2.  **Intelligent Query Generation:** An Anthropic Claude LLM analyzes your request and generates multiple optimized search queries to explore Google Custom Search effectively.
3.  **Initial Search & LLM Reranking:** The agent executes these queries on Google Custom Search. It then uses the Claude LLM to rapidly assess the titles and snippets of the initial search results, **intelligently filtering down to only the most promising links.**
4.  **Detailed Scraping & LLM Assessment:** Only the *highly promising* links are then visited to scrape detailed information. This extracted data is sent to the Claude LLM for a comprehensive assessment, including resource type classification, relevance, skill level, quality ratings, a concise summary, **and a transparent reasoning.**
5.  **Smart Ranking & Categorization:** A custom scoring algorithm combines the LLM's insights with your preferences to rank the resources. The resources are then grouped into logical categories for clear presentation.
6.  **Interactive Report:** The top curated resources are presented in a beautifully formatted, categorized HTML report, displayed directly within the Gradio interface.

## ⚙️ Local Setup and Running

Follow these steps to get the Personalized Learning AI Recommender running on your local machine.

### Prerequisites

* Python 3.8+
* An Anthropic API Key (you can get one from the [Anthropic Console](https://console.anthropic.com/))
* A Google Cloud Project with the Custom Search API enabled and a Programmable Search Engine configured (see [Google Custom Search API setup](https://developers.google.com/custom-search/v1/overview) for details).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/personalized-learning-recommender.git](https://github.com/your-username/personalized-learning-recommender.git)
    cd personalized-learning-recommender
    ```
    (Replace `your-username/personalized-learning-recommender.git` with your actual GitHub repository URL.)

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat
        ```
    * **Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up your API Keys:**
    Create a file named `.env` in the root directory of your project (the same folder where `app.py` is located).
    Add your API keys to this file:
    ```
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    GOOGLE_CSE_ID="your_google_custom_search_engine_id_here"
    ```
    **Important:** Never share this `.env` file or your API keys publicly. It's already included in `.gitignore` to prevent accidental commits.

2.  **Database Cache Reset (Important for Schema Changes):**
    If you've previously run the application, the `cache.db` file might have an older database schema. To ensure the new `category` and `reasoning` fields are correctly added, delete the old cache file:
    ```bash
    rm cache.db  # macOS/Linux
    # del cache.db # Windows
    ```
    The `init_db()` function will recreate it with the correct schema when the app starts.

### Running the Application

1.  **Start the Gradio Web UI:**
    Ensure your virtual environment is active.
    ```bash
    python app.py
    ```
2.  The application will start, and you'll see a local URL (e.g., `http://127.0.0.1:7874/`) and a temporary public shareable URL printed in your terminal. Open either of these in your web browser.

## 📂 Project Structure
.
├── app.py                  # Main application logic and Gradio UI
├── requirements.txt        # Python dependencies
├── .env.example            # Template for .env (can be added optionally for clarity)
├── .gitignore              # Specifies files and folders to ignore in Git
├── README.md               # This file!
├── cache.db                # SQLite database for caching (generated at runtime, ignored by Git)
├── logs/                   # Directory for application logs (generated at runtime, ignored by Git)
├── reports/                # Directory for HTML reports (generated at runtime, ignored by Git)
└── venv/                   # Python virtual environment (ignored by Git)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/your-username/personalized-learning-recommender/issues).

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
*Generated with 🧡 by an AI Coding Partner*
