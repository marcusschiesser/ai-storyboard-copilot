import replicate
import json
import os
import logging
from dotenv import load_dotenv
import requests
from pathlib import Path
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add this constant at the top of the file, after the imports
NR_IMAGES = 5  # Number of images to generate per scene


def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    with open(file_path, "r") as file:
        return json.load(file)


def call_replicate_api(prompt):
    prompt = f"{prompt}. style: ancient rome 0 ad"
    logging.info(f"Calling replicate api with prompt: {prompt[:50]}...")
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


def generate_prompt(description):
    logging.info(f"Generating prompt for description: {description[:50]}...")
    prompt = f"Generate a detailed and unique image prompt based on the following scene description: {description}. The prompt should be suitable for an image generation AI model. Style: Ancient Rome 0 AD. Make each prompt distinct and creative."

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",  # Update this to the correct model name
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant specialized in creating detailed and diverse image prompts. Just output the prompt and nothing else.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
    )

    generated_prompt = response.choices[0].message.content.strip()
    logging.info(f"Generated prompt: {generated_prompt[:50]}...")
    return generated_prompt


def generate_images(data):
    results = []
    total_scenes = len(data["scenes"])
    for i, scene in enumerate(data["scenes"], 1):  # This already starts at 1
        logging.info(f"Processing scene {i} ({i}/{total_scenes})")
        scene_results = {"scene_number": i, "images": []}

        for j in range(NR_IMAGES):
            # Generate a unique prompt for each image
            prompt = generate_prompt(scene["description"])

            # Generate image using the prompt
            logging.info(f"Generating image {j+1}/{NR_IMAGES} for scene {i}")
            result = call_replicate_api(prompt)

            logging.info(f"Generated image: {result}")

            image_url = result[0]
            filename = f"scene_{i}_image_{j+1}.webp"
            download_image(image_url, "output", filename)
            scene_results["images"].append(
                {
                    "version": f"v{j+1}",
                    "url": image_url,
                    "filename": filename,
                    "prompt": prompt,
                }
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
