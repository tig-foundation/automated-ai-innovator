from autoinnovator import AutoInnovatorBase, Challenge, Candidate, Context, LLM, LLMProvider
import dotenv
import os

dotenv.load_dotenv()
if not (API_KEY := os.getenv("API_KEY")):
    raise ValueError("API_KEY environment variable is not set. Edit .env file with your OpenAI API key.")

challenge = Challenge(
    name="kde",
    config_id="2d"
)

system_prompt = f"""
Code up an algorithm in Python 3 with the following goals and specifications:
```
Write a function that takes in training data points of shape (num_train_pts, dims) and returns Gaussian mixture model parameters: component unnormalised weight logits of shape (num_comps,), component means (num_comps, dims) and component covariances (num_comps, dims, dims). 
You are implementing kernel density estimation using some form of bandwidth selection and kernel placement heuristic.
Use only NumPy and implement the rest from scratch.
```

I will run your algorithm on a set of problem instances and report back to you the algorithm code I ran with your provided reasoning of writing this algorithm as well as evaluation results feedback.

Problem instances are generated with the following parameters: 
```
{challenge.parameters}
```

All prompts I send you thus contains three parts marked by "ALGORITHM:", "REASONING:" and "EVALUATION:".

The evaluation feedback contains:
```
Target score to maximize is test likelihoods (Gaussian mixture model evaluated on num_test_pts test data points) on a fixed set of problem instances (different datasets, here we test on 64 instances), we provide both per instance likelihoods and a single average over instances. In addition, you will get:
- Train data likelihoods (per instance and averaged)
- Time to run evaluation for all instances (we want this to be as low as possible, but prioritize the target score)
- Code complexity score (length of the Python code in bytes, we prefer algorithms where this is not too high)
You can come up with your own internal objective function (e.g. average test likelihood penalised by code complexity or so)
```

If there is a bug, you will instead receive the corresponding error from the Python interpreter.
We keep iterating to improve your candidate algorithm target score. Keep your responses short: first part is only your code annotated with comments to explain where necessary; second part is your summary of changes and your reasoning/thoughts on why you chose those.
For your response, adhere to the format: 
```
Format your output using XML tags as "<python>Code here</python><reasoning>Reasoning here</reasoning>", where you output code between "<python>" tags such that I can simply cut out the parts between the python tag and write it directly into some Python script file that I can use to import your function as a symbol into the evaluation script, and similarly your reasoning/thoughts/additional metadata into "<reasoning>" tags.

In particular, you should include notes of ideas you tried that worked and did not work well in your reasoning response, as I will keep feeding this back your latest code modifications and you can remember why you made a particular change based on past experience to avoid making the same mistake or modification twice.
```

I start with running the base algorithm implementation: 
```
{challenge.base_algorithm}
```

Then I will report back to you the evaluation feedback in the prompt format as discussed previously.
"""

llm = LLM(
    provider=LLMProvider.OPENAI,
    model="gpt-4.1-mini-2025-04-14",
    api_key=API_KEY
)

class SimpleAutoInnovator(AutoInnovatorBase):
    def create_prompt_kwargs(self, candidate: Candidate, ctx: Context) -> dict:
        if candidate.generation == 1:
            prev_candidate = ctx.candidates[0][0] # 0th gen just has 1 candidate using base algorithm
        else:
            prev_candidate = ctx.candidates[candidate.generation - 1][candidate.id] # just use 1 candidate from previous generation
        algorithm = prev_candidate.algorithm
        response = prev_candidate.response
        if response:
            prev_response_id = response["id"]
            text = response["output"][0]["content"][0]["text"]
            reasoning = text.split('<reasoning>')[1].split('</reasoning>')[0]
        else:
            prev_response_id = None
            reasoning = "No reasoning yet. Evaluation was on base algorithm."
        evaluation = prev_candidate.evaluation
        evaluation["algorithm_code_length"] = len(algorithm)
        
        return {
            "instructions": system_prompt,
            "input": f"ALGORITHM:\n{algorithm}\nREASONING:\n{reasoning}\nEVALUATION:\n{evaluation}",
            "temperature": 1.0,
            "previous_response_id": prev_response_id,
        }

    def extract_algorithm_code(self, response: dict) -> str:
        text = response["output"][0]["content"][0]["text"]
        if "<python>" in text and "</python>" in text:
            return text.split("<python>")[1].split("</python>")[0].strip()
        raise ValueError("Response does not contain valid algorithm code.")
    
innovator = SimpleAutoInnovator(
    challenge=challenge,
    llm=llm,
)

innovator.run(
    num_generations=10,
    candidates_per_generation=1,
    experiment_dir="example_experiment",
    num_visualisations_per_candidate=4
)