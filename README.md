# AI-Powered Scene Image Generator

This project generates images based on scene descriptions using AI models.

## Features

- Loads scene descriptions from a JSON file
- Generates unique prompts for each scene using GPT-4
- Creates multiple images per scene using Replicate's image generation API
- Downloads and saves generated images

## Prerequisites

- Python 3.8+
- Poetry

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   poetry install
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     REPLICATE_API_TOKEN=your_replicate_api_token
     OPENAI_API_KEY=your_openai_api_key
     ```
   - Use `REPLICATE_MODEL` to specify the model to use for image generation. If not set, the [default model Flux-Pro](https://replicate.com/black-forest-labs/flux-pro) will be used. 
   - If it's a Lora model add `USE_LORA=true` to the .env file

## Usage

1. Prepare your scene descriptions in `data.json`
2. Run the script:
   ```
   poetry run python main.py
   ```
3. Generated images will be saved in the `output` folder

## Customization

- Adjust the number of images per scene by modifying the `NR_IMAGES` constant in `main.py`
- Modify the image generation parameters in the `call_replicate_api` function
- Change the style or theme by updating the prompt in the `generate_prompt` function

## License

[MIT License](LICENSE)