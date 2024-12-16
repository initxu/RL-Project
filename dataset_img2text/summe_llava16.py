from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

import os
import json

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
device=0
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to(f"cuda:{device}")


# prompt
prompt_ls=[
    "You are an expert in visual content analysis. Each image you receive is a single frame from a video sequence. Analyze the frame and generate a concise and descriptive caption, clearly indicating the key elements and actions visible in the image. Consider the broader context of the video and ensure the caption contributes to the overall narrative.", 
    "You are an expert in visual content analysis. Each image you receive is a single frame from a video sequence. Analyze the frame and generate a concise and descriptive caption.",
    "What is shown in this image?",
    "What is shown in this image? Please give me a caption less than 30 words.",
]
prompt_id=2
prompt=prompt_ls[prompt_id]
print(f'******prompt: {prompt}******\n')

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": prompt_ls[prompt_id]}, 
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

dataset_img_folder='../dataset_img/summe_frames_downsample/'
result_file='./summe/llava-v1.6-mistral-7b-hf/v{}/{}.json'
video_ls=[
    'Air_Force_One', 'Base_jumping', 'Bearpark_climbing', 'Bike_Polo', 'Bus_in_Rock_Tunnel',\
        'Car_railcrossing', 'Cockpit_Landing', 'Cooking', 'Eiffel_Tower', 'Excavators_river_crossing', \
        'Fire_Domino', 'Jumps', 'Kids_playing_in_leaves', 'Notre_Dame', 'Paintball', \
        'Playing_on_water_slide', 'Saving_dolphines', 'Scuba', 'St_Maarten_Landing', 'Statue_of_Liberty',\
        'Uncut_Evening_Flight', 'Valparaiso_Downhill', 'car_over_camera', 'paluma_jump', 'playing_ball']


for video_name in video_ls:
    video_folder=os.path.join(dataset_img_folder,video_name)
    frame_ls=sorted(os.listdir(video_folder))
    results=[]
    for frame in frame_ls:
        image_file = os.path.join(video_folder,frame)
        print(f'process {image_file}')        
        raw_image = Image.open(image_file)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output_text=processor.decode(output[0][2:], skip_special_tokens=True)
        img_caption=output_text.split('[/INST] ')[1]
        print(img_caption)
        results.append({
            'frame':frame,
            'output_text':output_text,
            'img_caption':img_caption
        })
    with open(result_file.format(prompt_id,video_name), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False,)
