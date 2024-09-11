import replicate
import json
import os
import logging
from dotenv import load_dotenv
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    with open(file_path, "r") as file:
        return json.load(file)


def call_replicate_api(prompt):
    prompt = f"{prompt}. Style: Ancient rome 0 Ad"
    logging.info(f"Calling Replicate API with prompt: {prompt[:50]}...")
    output = replicate.run(
        "marcusschiesser/flux-dev-me:5cbdafc09fe365d8ffadf39308246f2015a1774afc86ae9bf0d1717a77404352",
        input={
            "model": "dev",
            "prompt": prompt,
            "lora_scale": 1,
            "num_outputs": 1,
            "aspect_ratio": "16:9",
            "output_format": "webp",
            "guidance_scale": 3.5,
            "output_quality": 80,
            "prompt_strength": 0.8,
            "extra_lora_scale": 0.8,
            "num_inference_steps": 28,
        },
    )
    logging.info("Replicate API call completed")
    return output


def download_image(url, folder, filename):
    Path(folder).mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder, filename), "wb") as file:
            file.write(response.content)
        logging.info(f"Image saved: {filename}")
    else:
        logging.error(f"Failed to download image: {url}")


def generate_images(data):
    results = []
    total_scenes = len(data["scenes"])
    for i, scene in enumerate(data["scenes"], 1):
        logging.info(f"Processing scene {scene['scene_number']} ({i}/{total_scenes})")
        scene_results = {"scene_number": scene["scene_number"], "images": []}
        for version, prompt in scene["prompts"].items():
            logging.info(f"Generating image for version {version}")
            result = call_replicate_api(prompt)
            image_url = result[0]
            filename = f"scene_{scene['scene_number']}_{version}.webp"
            download_image(image_url, "output", filename)
            scene_results["images"].append(
                {"version": version, "url": image_url, "filename": filename}
            )
        results.append(scene_results)
    return results


def main(json_file_path):
    logging.info("Starting image generation process")
    load_dotenv()
    if "REPLICATE_API_TOKEN" not in os.environ:
        logging.error("REPLICATE_API_TOKEN environment variable is not set")
        raise Exception("REPLICATE_API_TOKEN environment variable is not set")

    data = load_data(json_file_path)
    results = generate_images(data)
    logging.info("Image generation process completed")
    return results


if __name__ == "__main__":
    results = main("data.json")
    print(json.dumps(results, indent=2))
