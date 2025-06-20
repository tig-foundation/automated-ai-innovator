from autoinnovator import AutoInnovatorBase, Challenge, Candidate, Context, LLM, LLMProvider
import os

class SimpleAutoInnovator(AutoInnovatorBase):
    @property
    def system_prompt(self) -> str:
        with open(os.path.join("example", "system_prompts", f"{self.challenge.name}.txt"), "r") as f:
            prompt = f.read()
        return prompt.format(
            parameters=self.challenge.parameters,
            base_algorithm=self.challenge.base_algorithm
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
            "instructions": self.system_prompt,
            "input": f"ALGORITHM:\n{algorithm}\nREASONING:\n{reasoning}\nEVALUATION:\n{evaluation}",
            "temperature": 1.0,
            "previous_response_id": prev_response_id,
        }

    def extract_algorithm_code(self, response: dict) -> str:
        text = response["output"][0]["content"][0]["text"]
        if "<python>" in text and "</python>" in text:
            return text.split("<python>")[1].split("</python>")[0].strip()
        raise ValueError("Response does not contain valid algorithm code.")
    

if __name__ == "__main__":
    import dotenv
    import argparse
    
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Simple Auto Innovator Example")
    parser.add_argument("challenge", type=str, choices=["kde", "clustering", "binning"], help="Name of the challenge")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY"), help="Your OpenAI API key. Can also be set via the API_KEY environment variable.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key not provided. Either set `--api-key <API_KEY> or edit .env file")

    challenge = Challenge(args.challenge)
    llm = LLM(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1-mini-2025-04-14",
        api_key=args.api_key,
    )
    innovator = SimpleAutoInnovator(
        challenge=challenge,
        llm=llm,
    )
    innovator.run(
        num_generations=10,
        candidates_per_generation=1,
        experiment_dir=os.path.join("experiments", args.challenge),
        num_visualisations_per_candidate=4
    )