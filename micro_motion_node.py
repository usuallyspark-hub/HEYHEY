import torch
import math
import cv2
import numpy as np

class MicroMotion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_count": ("INT", {"default": 48, "min": 8, "max": 240}),
                "breath_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "blink_chance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0}),
                "vibration": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 3.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "Animation/Organic"

    def animate(self, image, frame_count, breath_intensity, blink_chance, vibration):
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w, _ = img.shape
        frames = []

        for f in range(frame_count):
            t = f / frame_count
            dy = int(math.sin(t * 2 * math.pi) * breath_intensity)  
            vibr_x = int(math.sin(t * 40) * vibration)
            vibr_y = int(math.cos(t * 37) * vibration)

            frame = cv2.warpAffine(img, np.float32([[1,0,vibr_x],[0,1,dy+vibr_y]]), (w,h))

            if np.random.rand() < blink_chance and 0.1 < t < 0.9:
                blink_factor = np.random.uniform(0.1,0.6)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame[:,:,2] = (frame[:,:,2] * blink_factor).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

            frames.append(torch.from_numpy(frame.astype(np.float32)/255.0))

        return (torch.stack(frames),)
