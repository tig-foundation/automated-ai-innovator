from autoinnovator import AutoInnovatorBase, Candidate, Context
import os

class SimpleAutoInnovator(AutoInnovatorBase):
    def system_prompt(self, ctx: Context) -> str:
        with open(os.path.join("example_prompts", f"{ctx.challenge.name}.txt"), "r") as f:
            prompt = f.read()
        return prompt.format(
            parameters=ctx.challenge.parameters
        )

    def create_prompt_kwargs(self, candidate: Candidate, ctx: Context) -> dict:
        if candidate.generation == 1:
            prev_candidate = ctx.candidates[0][0] # 0th gen just has 1 candidate using base algorithm
        else:
            prev_candidate = ctx.candidates[candidate.generation - 1][candidate.id] # just use 1 candidate from previous generation
        if not prev_candidate.success:
            # Another strategy would be to use the previous generation's best candidate
            raise RuntimeError("Previous candidate was not successful. Simple Innovator strategy cannot continue.")
        algorithm = prev_candidate.algorithm
        response = prev_candidate.response
    
        text = response["choices"][0]["message"]["content"] if response else None

        if text:
            prev_response_id = response["id"]
            reasoning = text.split('<reasoning>')[1].split('</reasoning>')[0]
        else:
            prev_response_id = None
            reasoning = "No reasoning yet. Evaluation was on base algorithm."
        evaluation = prev_candidate.evaluation
        evaluation["algorithm_code_length"] = len(algorithm)
        
        prompt = {
            "messages": [
                {"role": "system", "content": self.system_prompt(ctx)},
                {"role": "user", "content": f"ALGORITHM:\n{algorithm}\nREASONING:\n{reasoning}\nEVALUATION:\n{evaluation}"}
            ],
            "temperature": 1.0,
            "previous_response_id": prev_response_id,
        }

        return prompt

    def extract_algorithm_code(self, response: dict, ctx: Context) -> str:
        text = response["choices"][0]["message"]["content"] if response else None

        if text and "<python>" in text and "</python>" in text:
            return text.split("<python>")[1].split("</python>")[0].strip()
        raise ValueError("Response does not contain valid algorithm code.")

API_KEY = None
if not API_KEY:
    raise ValueError("You must set an API Key")

from autoinnovator import Challenge, LLM, LLMProvider
llm = LLM(
    provider=LLMProvider.AKASH,
    model="DeepSeek-R1-Distill-Llama-70B",
    api_key=API_KEY
)
my_autoinnovator = SimpleAutoInnovator()
kde_challenge = Challenge("kde")
binning_challenge = Challenge("binning")
clustering_challenge = Challenge("clustering")

from IPython.display import Image, display, clear_output, Markdown
import time

num_visualisations = 4
def on_generation_done(ctx: Context):
    clear_output(wait=True)
    best_candidate = max(
        filter(lambda c: c.success, ctx.candidates[ctx.curr_generation]),
        key=lambda c: c.evaluation["test_log_likelihood_average"]
    )
    display(Markdown(f"### Generation {ctx.curr_generation} - Best Candidate {best_candidate.id}"))
    for i in range(num_visualisations):
        display(Image(filename=best_candidate.visualisation_path.format(i=i)))
    display(Image(filename=ctx.results_plot_path))
    time.sleep(1)

kde_ctx = my_autoinnovator.run(
    llm=llm,
    challenge=kde_challenge,
    num_generations=3,
    candidates_per_generation=1,
    experiment_dir=os.path.join("experiments", "kde"),
    num_visualisations_per_candidate=num_visualisations,
    on_generation_done=on_generation_done
)