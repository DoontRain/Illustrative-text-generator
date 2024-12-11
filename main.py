from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from diffusers import FluxPipeline
import torch
from PIL import Image
import base64
import io
import os
from huggingface_hub import login
login(token="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")
app = Flask(__name__)

# 配置OpenAI API客户端
openai_client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-uEvWN0eIkjRrG4HZ2FGVSa2WkAmqRDTJeL6R41l2D3hqj8hL'
)
print("Flux model...")
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")


# 加载本地模型
# import torch

# pipe = FluxPipeline.from_single_file(
#     pretrained_model_link_or_path="./model/flux1-dev-fp8.safetensors",
#     torch_dtype=torch.float16,
#     safety_checker=None,
#     device="cuda",
#     low_cpu_mem_usage=True
# ).to("cuda")
# #
# # 最大程度优化显存
# pipe.enable_attention_slicing(1)
# pipe.enable_model_cpu_offload()  # 比 sequential_cpu_offload 更激进的显存优化
# pipe.enable_vae_slicing()
# pipe.enable_vae_tiling()
# torch.cuda.empty_cache()  # 清理 GPU 缓存
# pipe = FluxPipeline.from_pretrained(
#     "blackforest",
#     use_auth_token=True  # 确保使用你的 Hugging Face token 进行授权
# )

# 如果有GPU则使用GPU
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("Using CUDA")
else:
    print("Warning: CUDA not available, using CPU. This will be slow!")


def generate_story(prompt):
    """使用OpenAI API生成短文"""
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "你是一个创意写手，请根据关键词写一段简短的富有画面感的描述，大约50-100字。同时在结尾加上这段描述的英文翻译，用于生成图像。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


def generate_image(text_prompt):
    """使用本地Stable Diffusion模型生成图像"""
    try:
        # 提取英文提示词（假设在文本最后一段）
        prompt_lines = text_prompt.split('\n')
        english_prompt = prompt_lines[-1].strip()

        print(f"Generating image with prompt: {english_prompt}")

        # 生成图像
        with torch.inference_mode():
            image = pipe(
                prompt=english_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                negative_prompt="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
            ).images[0]

        # 转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str
    except Exception as e:
        print(f"Error in image generation: {str(e)}")
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']

    try:
        # 生成文本
        story = generate_story(prompt)
        print(f"Generated story: {story}")

        # 生成图像
        image = generate_image(story)

        return jsonify({
            'story': story,
            'image': image
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)