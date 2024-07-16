import os
import random
import time
import urllib
from collections import Counter

import cv2
import pandas as pd
import requests
from openai import OpenAI


class ImageGenerator:

    def __init__(self, prompt, file_location, count):
        self.prompt = prompt
        self.file_location = file_location
        self.count = count
        self.time_ = int(time.time())
        self.stable_diffusion_name = None
        self.dall_3_name = None
        self.header = ['image', 'prompt']
        self.stable_diffusion_df = pd.DataFrame(columns=self.header)
        self.dall_3_df = pd.DataFrame(columns=self.header)
        self.df = pd.DataFrame(columns=['stable_diffusion_image', 'dall_3_image', 'prompt'])

    def resize_image(self, original_i, resized_i):
        img = cv2.imread(original_i)
        original_height, original_width = img.shape[:2]

        target_ratio = 2448 / 2048

        current_ratio = original_width / original_height

        if current_ratio == target_ratio:
            cropped_img = img
        elif current_ratio > target_ratio:
            new_width = int(original_height * target_ratio)
            offset = (original_width - new_width) // 2
            cropped_img = img[:, offset:offset + new_width]
        else:
            new_height = int(original_width / target_ratio)
            offset = (original_height - new_height) // 2
            cropped_img = img[offset:offset + new_height, :]

        resized_image = cv2.resize(cropped_img, (2448, 2048), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(resized_i, resized_image)

    def stable_diffusion(self):
        print("stable_diffusion")
        api_key = ''  # add your own key
        negative_prompt = ""  # @param {type:"string"}
        aspect_ratio = "5:4"  # @param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
        seed = 0  # @param {type:"integer"}
        output_format = "jpeg"  # @param ["jpeg", "png"]

        host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {api_key}"
        }
        files = {"none": ''}
        params = {
            "prompt": self.prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "output_format": output_format,
            "model": "sd3",
            "mode": "text-to-image"
        }

        response = requests.post(
            host,
            headers=headers,
            files=files,
            data=params
        )
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        # Decode response
        output_image = response.content
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        # Check for NSFW classification
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")

        # Save and display result
        generated_name = f"{self.count}_stable_diffusion_{self.time_}_{seed}.{output_format}"
        generated = f"{self.file_location}/{generated_name}"
        with open(generated, "wb") as f:
            f.write(output_image)
        print(f"Saved image {generated}")

        # resize_image
        resized_image = generated.split('.')[0] + f"_resized.{output_format}"
        self.resize_image(generated, resized_image)
        print(f"Resize image {resized_image}")

        self.stable_diffusion_name = generated_name

    def dall_3(self):
        print("dall_3")
        api_key = ''  # add your own key
        n = 1
        quality = "standard"  # hd, standard
        size = "1024x1024"  # 1024x1024, 1792x1024, 1024x1792
        output_format = "jpeg"

        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=n,
            quality=quality,
            response_format="url",
            size=size,
            style="vivid"
        )

        image_url = response.data[0].url

        generated_name = f"{self.count}_dall_3_{quality}_{self.time_}.{output_format}"
        generated = f"{self.file_location}/{generated_name}"
        urllib.request.urlretrieve(image_url, generated)
        print(f"Saved image {generated}")

        # resize_image
        resized_image = generated.split('.')[0] + f"_resized.{output_format}"
        self.resize_image(generated, resized_image)
        print(f"Resize image {resized_image}")

        self.dall_3_name = generated_name

    def write_to_csv(self):
        data = {"stable_diffusion_image": self.stable_diffusion_name, "dall_3_image": self.dall_3_name, "prompt": self.prompt}
        self.df = self.df.append(data, ignore_index=True)

    def save_to_csv_all(self):
        if not self.df.empty:
            file_path = "images_/all_laptop/image_prompt.csv"
            if os.path.isfile(file_path):
                self.df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                self.df.to_csv(file_path, index=False)
            self.df = pd.DataFrame(columns=['stable_diffusion_image', 'dall_3_image', 'prompt'])


if __name__ == '__main__':

    laptop_model_list = ["a Lenovo laptop", "a ThinkPad laptop", "an HP laptop", "a Dell laptop", "a MacBook laptop", "a laptop"]
    sticker_num_list = ["one", "two", "three"]
    sticker_size_list = ["small", "medium", "big"]
    sticker_style_list = ["cartoon character", "nature-themed", "animal", "motivational quote", "floral", "sports",
                          "abstract art", "holiday-themed", "emoji", "superhero", "vintage", "food", "travel",
                          "music", "fantasy", "Apple logo", "Dell logo", "HP logo", "ThinkPad logo", "Lenovo logo"]
    sticker_location_list = ["close to the center of the laptop", "close to the edge of the laptop",
                             "stickers spread out", "stickers close to each other", ""]

    prompt = ""
    file_location = "images"
    count = 0
    ig = ImageGenerator(prompt, file_location, count)

    if not os.path.exists("images"):
        os.makedirs("images")

    for i in range(count, count + 10):
        laptop_model = random.choice(laptop_model_list)
        sticker_num = random.choice(sticker_num_list)
        prompt = f"{laptop_model}, {sticker_num} {'sticker' if sticker_num == 'one' else 'stickers'} on the lid, "

        if sticker_num == "three":
            sticker_size_list = ["small"]
        elif sticker_num == "two":
            sticker_size_list = ["small", "medium"]
        else:  # "one"
            sticker_size_list = ["small", "medium", "big"]
        sticker_size_style_list = []
        for _ in range(sticker_num_list.index(sticker_num) + 1):
            sticker_size_style_list.append(
                f"one {random.choice(sticker_size_list)} {random.choice(sticker_style_list)} sticker")

        counter = Counter(sticker_size_style_list)
        number_words = {
            1: "one",
            2: "two",
            3: "three"
        }
        sticker_size_style_list_counter = []
        for item, count in counter.items():
            parts = item.split(' ', 1)
            new_item = f"{number_words[count]} {parts[1]}"
            sticker_size_style_list_counter.append((count, new_item))
        sticker_size_style_list_counter.sort(key=lambda x: x[0])
        sticker_size_style_list_sorted = [item[1] for item in sticker_size_style_list_counter]
        for sticker_size_style in sticker_size_style_list_sorted:
            prompt += sticker_size_style + ", "

        if sticker_num == "one":
            sticker_location_list = ["sticker close to the center of the laptop", "sticker close to the edge of the laptop", ""]
        else:
            sticker_location_list = ["stickers close to the center of the laptop", "stickers close to the edge of the laptop",
                                     "stickers spread out", "stickers close to each other", ""]
        sticker_location = random.choice(sticker_location_list)
        if sticker_location != "":
            prompt += f"{sticker_location}, "

        prompt += "top-down view, screen turn off, on a flat surface, a solid dark background"

        print(f"image {i + 1} {prompt}")
        ig.prompt = prompt
        ig.count += 1
        ig.time_ = int(time.time())
        ig.dall_3()
        ig.stable_diffusion()

        ig.write_to_csv()
        ig.save_to_csv_all()
