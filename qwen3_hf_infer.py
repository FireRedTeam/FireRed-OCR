import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
from tqdm import tqdm
from dataclasses import asdict
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, Qwen3VLForConditionalGeneration
import argparse
import torch
from conv_for_infer import generate_conv

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--processor_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def collect_images(input_dir):
    images = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, name))
    return images


def split_list(data, n):
    """均匀切分 list 到 n 份"""
    return [data[i::n] for i in range(n)]


def worker(
    rank,
    gpu_id,
    image_paths,
    model_dir,
    processor_dir,
    output_dir,
):
    # ⚠️ 必须在 import vllm 之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # from vllm import LLM, EngineArgs, SamplingParams

    print(f"[Worker {rank}] Using GPU {gpu_id}, images: {len(image_paths)}")

    processor = AutoProcessor.from_pretrained(processor_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto"
    )

    for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}"):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        markdown_file = os.path.join(output_dir, f"{basename}.md")

        data_dict = {
            "image_path": image_path
        }
        messages = generate_conv(data_dict)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(text)
        return text


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    assert len(image_paths) > 0, "No images found"

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices found"

    print(f"Detected {num_gpus} GPUs, total images: {len(image_paths)}")

    chunks = split_list(image_paths, num_gpus)

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(range(num_gpus), chunks)):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                gpu_id,
                chunk,
                args.model_dir,
                args.processor_dir,
                args.output_dir,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

"""
python qwen3_hf_infer.py \
    --model_dir /workspace/Qwen3-VL-2B-Instruct \
    --processor_dir /workspace/Qwen3-VL-2B-Instruct \
    --input_dir /workspace/cropped \
    --output_dir /workspace/outputs
"""