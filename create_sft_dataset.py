import os
import json
import asyncio
import aiofiles

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from Courses.STF_LLM.Assignment_5.utils.math_utils import r1_zero_reward_fn


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url=DEEPSEEK_BASE_URL,
)

init_prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Try to use more mathematical formulas in your thinking process.
IMPORTANT: The <answer> tag must contain only the final answer braced with \\boxed, with no extra words. For example: <answer>\\boxed{{-21}}</answer>. Please present the final answer in LaTeX format.
User: {question}
Assistant:"""

modi_prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Try to use more mathematical formulas in your thinking process.

User: {question}\nSolution: {solution}

Give you the solution process, please generate consice thinking process based on the solution. Place thinking process inside the `<think>` and `</think>` tags. Identify the final answer within the provided `solution`, just follow `\\boxed` format. Copy this final answer **exactly and verbatim** inside the `<answer>` and `</answer>` tags.

Assistant:"""


INPUT_FILE_PATH  = "./data/math/train.jsonl"
OUTPUT_FILE_PATH = "./data/math/sft.jsonl" 
WRONG_FILE_PATH  = "./data/math/wrong.jsonl"
CONCURRENCY_LIMIT = 20


async def process_single_problem(sem, line, index, outfile, wrong_file):
    """Async function to process a single problem"""
    async with sem:  # use semaphore to control concurrency
        try:
            data = json.loads(line.strip())
            problem  = data.get("problem")
            solution = data.get("solution")

            is_correct = False
            response = ""

            # initalize prompts with question and solution
            init_prompt = init_prompt_template.format(question=problem)
            modi_prompt = modi_prompt_template.format(question=problem, solution=solution)
            prompts = [init_prompt, modi_prompt]
            
            for attempt, current_prompt in enumerate(prompts):
                try:
                    chat_completion = await client.chat.completions.create(
                        model='deepseek-reasoner',
                        messages=[{"role": "user", "content": current_prompt}],
                        max_tokens=4096,
                        temperature=0.2,
                        timeout=60,
                    )
                    response = chat_completion.choices[0].message.content.strip()
                    
                    # evaluate the response (internally uses standardized grading)
                    result = r1_zero_reward_fn(response, solution, fast=True)
                    
                    # Log printing: the result will be messy because of concurrency
                    print(f"Problem {index} Attempt {attempt+1}: Reward {result.get('reward', 0.0)}")

                    if result.get('reward', 0.0) == 1.0:
                        is_correct = True
                        break # Break loop if successful
                        
                except Exception as e:
                    print(f"Error processing problem {index}, attempt {attempt+1}: {e}")
                    continue

            if is_correct:
                right_record = {"problem": problem, "response": response}
                await outfile.write(json.dumps(right_record, ensure_ascii=False) + '\n')
                await outfile.flush()
                return True
            else:
                wrong_record = {"problem": problem, "solution": solution, "response": response}
                await wrong_file.write(json.dumps(wrong_record, ensure_ascii=False) + '\n')
                await wrong_file.flush()
                return False

        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line at index {index}")
            return False


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    # semaphore to limit concurrency
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # read all lines from input file
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # asynchronously open output files
    async with aiofiles.open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as outfile, \
               aiofiles.open(WRONG_FILE_PATH,  'w', encoding='utf-8') as wrong_file:
        
        tasks = []        
        for i, line in enumerate(lines):
            # create tasks for each problem and run asyncio gather later
            task = asyncio.create_task(process_single_problem(sem, line, i + 1, outfile, wrong_file))
            tasks.append(task)
        
        # use tqdm to show progress
        results = await tqdm_asyncio.gather(*tasks, desc="Processing problems (Async)")
        
        success_count = sum(1 for r in results if r)
        print(f"\n🎉 Processing complete! Success: {success_count}, Failed: {len(results) - success_count}")

if __name__ == "__main__":
    asyncio.run(main())
