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
    * **Perception:** Intelligently performs web searches and scrapes diverse web content.
    * **Reasoning & Planning:** Uses Anthropic's Claude LLM to understand user needs, classify resources, assess quality, and determine relevance. It plans a multi-step process from search to curated output.
    * **Action:** Executes web requests, parses HTML, and generates a rich HTML report.
    * **Autonomy:** Operates autonomously from user input to final recommendations.
* **Enhanced Accuracy with LLMs:** Leverages Anthropic Claude for deep semantic understanding of learning content, providing more precise recommendations than rule-based systems.
* **Cost Efficiency & Performance:** Implements a local SQLite caching mechanism to minimize redundant web scraping and expensive LLM API calls.
* **User-Friendly Interface:** Provides an interactive web UI using Gradio, making it easy for anyone to use without command-line knowledge.
* **Production-Ready Practices:** Includes robust error handling, structured logging, environment variable management for API keys, and adherence to web scraping politeness.

## 🚀 How It Works

1.  **User Input:** You provide a learning goal (e.g., "Python for Data Science"), a preferred learning style (e.g., "video", "articles", "hands-on projects"), and your current skill level.
2.  **Intelligent Query Generation:** An Anthropic Claude LLM analyzes your request and generates multiple optimized search queries to explore the web effectively.
3.  **Web Search & Scraping:** The agent executes these queries using DuckDuckGo and then visits promising links to scrape detailed information (titles, descriptions, URLs).
4.  **LLM-Powered Curation:** For each unique resource, the agent checks its local cache. If not found or expired, it sends the resource's metadata to the Claude LLM for a comprehensive assessment, including:
    * Resource Type Classification (e.g., "course", "documentation")
    * Relevance Score (1-5)
    * Skill Level Match (e.g., "perfect", "good")
    * Overall Quality Rating (e.g., "excellent", "average")
    * A concise summary of the resource.
5.  **Smart Ranking & Filtering:** A custom scoring algorithm combines the LLM's insights with your preferences to rank the resources. Low-scoring or irrelevant resources are filtered out.
6.  **Interactive Report:** The top curated resources are presented in a beautifully formatted HTML report, displayed directly within the Gradio interface.

## ⚙️ Local Setup and Running

Follow these steps to get the Personalized Learning AI Recommender running on your local machine.

### Prerequisites

* Python 3.8+
* An Anthropic API Key (you can get one from the [Anthropic Console](https://console.anthropic.com/))

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/personalized-learning-recommender.git](https://github.com/your-username/personalized-learning-recommender.git)
    cd personalized-learning-recommender
    ```
    (Replace `your-username/personalized-learning-recommender.git` with your actual GitHub repository URL after you push the code.)

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

1.  **Set up your Anthropic API Key:**
    Create a file named `.env` in the root directory of your project (the same folder where `app.py` is located).
    Add your API key to this file:
    ```
    ANTHROPIC_API_KEY="your_actual_anthropic_api_key_here"
    ```
    **Important:** Never share this `.env` file or your API key publicly. It's already included in `.gitignore` to prevent accidental commits.

### Running the Application

1.  **Start the Gradio Web UI:**
    Ensure your virtual environment is active.
    ```bash
    python app.py
    ```
2.  The application will start, and you'll see a local URL (e.g., `http://127.0.0.1:7860/`) and a temporary public shareable URL printed in your terminal. Open either of these in your web browser.

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
