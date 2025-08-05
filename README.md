# Automated AI Innovator

Framework for an Automated AI Innovator that operates in a loop, where each iteration—called a **generation**—produces a set of **candidate algorithms**. The process for each candidate is:

1. **Create Prompt** – Your defined `create_prompt_kwargs` is used to construct a request to send to a Large Language Model (LLM)
   * See https://platform.openai.com/docs/api-reference/chat/create for what can be included in your request
   * See `Context` and `Candidate` classes in [autoinnovator/framework.py](autoinnovator/framework.py) for what historic data you can access
3. **Send Prompt to LLM** – The prompt is submitted to a LLM.
4. **Extract Algorithm Code** – Your defined `extract_algorithm_code` is used to extract algorithm code from the LLM’s response.
5. **Evaluate Algorithm** – The candidate algorithm is tested and scored.

Each candidate is generated independently but has access to the full history of prior generations. The goal is to design your AutoInnovator in a way that uses data from previous generations to guide the creation of higher-performing algorithms over generations.

## Whitepaper: Technical Overview

For a detailed technical overview of the design, motivation, and broader vision, please refer to our whitepaper:

[Automated Algorithmic Innovator MVP - The Innovation Game Labs, August 2025](./whitepaper.pdf)

Our whitepaper outlines:

* The motivation behind automated algorithmic innovation
* Design principles of the MVP pipeline
* Details of the mini-challenges used for evaluation
* Future directions for multi-agent and reinforcement learning-based extensions

### Recompiling

To recompile the LaTeX source of the paper:
```
cd whitepaper
pdflatex main.tex 
biber main
pdflatex main.tex
pdflatex main.tex
```

This will generate an updated `main.pdf` in the `whitepaper/` directory.

## Quick Start

We have prepared a `quick_start.ipynb` Jupyter notebook which you can run locally, via Google Colab, or via Akash.

**You will need to signup and create an API Key with one of the following providers:**

* OpenAI: https://platform.openai.com/api-keys
* Akash Chat: https://chatapi.akash.network/

### Local Setup

1. Clone the repository
   ```
   git clone https://github.com/tig-foundation/automated-ai-innovator
   cd automated-ai-innovator
   ```

2. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```

3. Install Jupyter Lab 
   
   Follow instructions: https://jupyterlab.readthedocs.io/en/4.4.x/getting_started/installation.html

4. Start Jupyter Lab
   ```
   jupyter lab
   ```

5. Open `quick_start.ipynb`

6. Edit Section `1.2 Set Global Parameters`

   **Note:** You will need a valid LLM API Key

7. Run all cells in the notebook

### Google Colab 

1. Open `quick_start.ipynb` in Google Colab

   <a href="https://colab.research.google.com/github/tig-foundation/automated-ai-innovator/blob/main/quick_start.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

2. Sign in with Google

3. Edit Section `1.2 Set Global Parameters`

   **Note:** You will need a valid LLM API Key

4. Run all cells in the notebook

### Akash Deployment

1. Download `deploy.yml` from this repo

2. Visit Akash console: https://console.akash.network/

3. Start Jupyter Lab with this repo

   Choose "Upload your SDL" and select `deploy.yml`

4. Access Jupyter Lab (token is `autoinnovator`)

   After deployment, goto `Leases -> Forwarded Ports` to find the url

5. Open `quick_start.ipynb`

6. Edit Section `1.2 Set Global Parameters`

   **Note:** You will need a valid LLM API Key

7. Run all cells in the notebook

## License

[Apache License 2.0](./LICENSE)