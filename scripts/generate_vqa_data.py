import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from config import CC3M_TRAIN_CSV_PATH, CC12M_TRAIN_CSV_PATH, DATA_DIR
from fuse_clip.fuse_clip_utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--bs", type=int, default=128, help="Batch size per GPU")
parser.add_argument("--validate", type=str, default=None)
parser.add_argument("--merge", action="store_true")
parser.add_argument("--cc12m", action="store_true")


SYSTEM_PROMPT = (
    "You are a helpful assistant, that generates a question and answer about an image that has a given caption. "
    "Rules:\n"
    "1. For generating question/answer pairs, only use information that is evident from the caption.\n"
    "2. Do not mention the word 'caption' in the question or answer.\n"
    "3. Answers should be at least a couple words long (not single word).\n"
    "4. Don't start every question with \"What\"\n"
    "5. Respond in the format: Question: <question>\nAnswer: <answer>\n"
    "Examples:\n"
    "Caption: A group of friends are having a barbecue in the backyard. Question: Where is the barbecue taking place?\n"
    "Answer: In the backyard.\n"
    "Caption: A child is playing with a toy airplane on the floor. Question: Which toy is the child holding?\n"
    "Answer: A toy airplane.\n"
    "Caption: A girl with a smartphone is lying on the sofa. Question: Where is the girl in the image located?\n"
    "Answer: On the sofa.\n"
    "Caption: A golden retriever is running through a field of flowers. Question: What is the dog doing?\n"
    "Answer: Running through a field of flowers.\n"
)

def extract_caption(output: Dict) -> str:
    msg = output["generated_text"][1]["content"]
    return msg.split("Caption: ")[1].strip()

def extract_question_answer(output: Dict) -> Tuple[str, str]:
    msg = output["generated_text"][2]["content"]
    question = msg.split("Question: ")[1].split("\nAnswer: ")[0].strip()
    answer = msg.split("\nAnswer: ")[1].strip()
    return question, answer


def process_output(output: Dict):
    caption = extract_caption(output)
    question, answer = extract_question_answer(output)
    return caption, question, answer

def save_data(questions: List[str], answers: List[str], image_paths: List[str], output_path: str):
    data = pd.DataFrame({
        "path": image_paths,
        "question": questions,
        "answer": answers,
    })
    data.to_csv(output_path, index=False)


@torch.inference_mode()
def run_inference_on_chunk(
    pipe,
    prompts_chunk: List[List[Dict[str, str]]],
    image_paths_chunk: List[str],
    batch_size: int,
    chunk_idx: int,
):
    """
    Run the pipeline on a chunk of data (prompts_chunk) using a given pipeline (pipe).
    Returns lists of questions, answers, and corresponding image_paths.
    """

    questions_chunk = []
    answers_chunk = []
    paths_chunk = []

    for batch_idx, start_idx in enumerate(tqdm(range(0, len(prompts_chunk), batch_size), disable=chunk_idx!=0)):
        prompts_batch = prompts_chunk[start_idx : start_idx + batch_size]

        # Perform generation
        generations = pipe(
            prompts_batch,
            batch_size=batch_size,
            do_sample=True,
            temperature=1.0,
            top_p=1,
            max_new_tokens=50,
            padding="longest",
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        # Collect results
        for i_sample, generation in enumerate(generations):
            # each 'generation' should be a list of length 1
            # but pipeline typically returns list-of-lists at top level
            assert len(generation) == 1
            try:
                caption, question, answer = process_output(generation[0])
            except Exception as e:
                # print(f"Failed inference with exception: {e}"
                #       f"Generation: {generation}")
                continue

            questions_chunk.append(question)
            answers_chunk.append(answer)
            paths_chunk.append(image_paths_chunk[start_idx + i_sample])

        if batch_idx % 10 == 0:
            partial_save_path = os.path.join(os.path.dirname(output_path), f"partial_chunk{chunk_idx}.csv")
            save_data(questions=questions_chunk, answers=answers_chunk,
                      image_paths=paths_chunk, output_path=partial_save_path)

    return questions_chunk, answers_chunk, paths_chunk


def merge_chunks(data_dir: str):
    dfs = []
    for el in os.listdir(data_dir):
        if el.startswith("partial_chunk"):
            data_path = os.path.join(data_dir, el)
            df_cur = pd.read_csv(data_path)
            dfs.append(df_cur)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(data_dir, "merged.csv"), index=False)



def validate_data(data_path: str):
    df = pd.read_csv(data_path)
    # check no duplicates in path
    assert len(df) == len(df["path"].unique())
    logging.info(f"{len(df)} rows in {data_path}")





if __name__ == '__main__':
    set_seed(0)
    logging.basicConfig(level=logging.INFO)

    # if --validate, only validate the data and exit
    args = parser.parse_args()
    print(args)
    cc_str = "cc12m" if args.cc12m else "cc3m"
    if args.validate is not None:
        validate_data(args.validate)
        logging.info(f"Validation successful for {args.validate}")
        exit(0)
    # if --merge, merge the data, then validate and exit
    if args.merge:
        data_dir = f"/mnt/cschlarmann37/data/{cc_str}-vqa/"
        merge_chunks(os.path.dirname(data_dir))
        logging.info(f"Merging successful")
        validate_data(os.path.join(data_dir, "merged.csv"))
        logging.info(f"Validation successful for {os.path.join(data_dir, 'merged.csv')}")
        exit(0)

    model_id = args.model
    batch_size = args.bs
    output_path = (f"{DATA_DIR}/{cc_str}-vqa/{model_id.split('/')[-1].replace('.', '-')}"
                   f"/{cc_str}_vqa_train_v2_.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_data_path = CC12M_TRAIN_CSV_PATH if args.cc12m else CC3M_TRAIN_CSV_PATH
    logging.info(f"Loading data from: {base_data_path}")
    base_data = pd.read_csv(base_data_path)
    captions = base_data["caption"].tolist()
    image_paths_base = base_data["path"].tolist()
    del base_data  # free up memory

    base_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    logging.info("Building list of prompts...")
    prompts = []
    for caption in tqdm(captions, desc="Generating prompts"):
        prompt_cur = base_prompt.copy()
        prompt_cur.append({"role": "user", "content": f"Caption: {caption}"})
        prompts.append(prompt_cur)

    prompts = prompts#[:10000]

    # Count number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available!")

    logging.info(f"Found {num_gpus} GPUs. Creating one pipeline per GPU.")

    # -------------------------------
    # Create one pipeline per GPU
    # -------------------------------
    pipes = []
    for gpu_id in range(num_gpus):
        logging.info(f"Initializing pipeline for GPU {gpu_id}...")
        pipe = pipeline(
            model=model_id,
            torch_dtype=torch.bfloat16,
            device=gpu_id  # replicate model to GPU= gpu_id
        )
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.tokenizer.padding_side = "left"
        pipes.append(pipe)

    # -------------------------------
    # Split data among GPUs
    # -------------------------------
    def chunk_list(lst, n):
        """Split list `lst` into `n` as evenly sized chunks as possible."""
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    prompts_chunks = chunk_list(prompts, num_gpus)
    paths_chunks = chunk_list(image_paths_base, num_gpus)

    # For multi‐threaded parallelism:
    #   we submit one "run_inference_on_chunk" per GPU
    #   each pipeline runs on its assigned data chunk
    logging.info("Starting multi-GPU inference...")

    all_questions = []
    all_answers = []
    all_paths = []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for chunk_idx, gpu_id in enumerate(range(num_gpus)):
            futures.append(
                executor.submit(
                    run_inference_on_chunk,
                    pipes[gpu_id],            # pipeline for GPU #gpu_id
                    prompts_chunks[gpu_id],   # chunk of prompts
                    paths_chunks[gpu_id],     # matching chunk of image paths
                    batch_size,
                    chunk_idx=chunk_idx
                )
            )

        # Collect all the results
        for fut in futures:
            questions_chunk, answers_chunk, paths_chunk = fut.result()
            logging.info(f"Chunk processed. Saving results...")
            all_questions.extend(questions_chunk)
            all_answers.extend(answers_chunk)
            all_paths.extend(paths_chunk)

    # Save final results
    logging.info(f"Saving output to {output_path}")
    save_data(all_questions, all_answers, all_paths, output_path)
    logging.info("All done!")
