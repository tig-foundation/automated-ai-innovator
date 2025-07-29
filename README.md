# Automated AI Innovator

Framework for an Automated AI Innovator that operates in a loop, where each iteration—called a **generation**—produces a set of **candidate algorithms**. The process for each candidate is:

1. **Create Prompt** – Your defined `create_prompt_kwargs` is used to construct a request to send to a Large Language Model (LLM)
   * See https://platform.openai.com/docs/api-reference/chat/create for what can be included in your request
   * See `Context` and `Candidate` classes in [autoinnovator/framework.py](autoinnovator/framework.py) for what historic data you can access
3. **Send Prompt to LLM** – The prompt is submitted to a LLM.
4. **Extract Algorithm Code** – Your defined `extract_algorithm_code` is used to extract algorithm code from the LLM’s response.
5. **Evaluate Algorithm** – The candidate algorithm is tested and scored.

Each candidate is generated independently but has access to the full history of prior generations. The goal is to design your AutoInnovator in a way that uses data from previous generations to guide the creation of higher-performing algorithms over generations.

## Quick Start

<a href="https://colab.research.google.com/github/tig-foundation/automated-ai-innovator/blob/akash-chat/quick_start.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## License

[Apache License 2.0](./LICENSE)