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

dataset_img_folder='../dataset_img/tvsum_frames_downsample/'
result_file='./tvsum/llava-v1.6-mistral-7b-hf/v{}/{}.json'
video_ls=['AwmHb44_ouw','akI8YFjEmUw','i3wAGJaaktw','Bhxk-O1Y7Ho','0tmA_C6XwfM','3eYKfiOEJNs','xxdtq8mxegs','WG0MBPpPC6I','Hl-__g2gn_A','Yi4Ij2NM7U4','37rzWOQsNIw','98MoyGZKHXc','LRw_obCPUt0','cjibtmSLxQ4','b626MiF1ew4','XkqCExn6_Us','GsAD1KT1xo8','PJrm840pAUI','91IHQYk1IQM','RBCABdttQmI','z_6gVvQb2d0','fWutDQy1nnY','J0nA4VgnoCo','4wU_LUjG5Ic','VuWGsYPqAX8','JKpqYvAdIsw','xmEERLqJ2kU','byxOvuiIJV0','_xMr-HKMfVA','WxtbjNsCQ8A','uGu_10sucQo','EE-bNr36nyA','Se3oxnaPsz0','gzDbaEs1Rlg','oDXZc0tZe04','qqR6AEXwxoQ','EYqVtI9YWJA','eQu1rNs0an0','JgHubY5Vw3Y','iVt07TCkFM0','E11zDS9XGzg','NyBmCxDoHJU','kLxoNp-UchI','jcoYJXDG9sw','XzYM3PfTM4w','-esJrBWj2d8','HT5vyqe0Xaw','sTEELN-vY30','vdmoEJ5YbrQ','xwqBXPGE9pQ']


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
