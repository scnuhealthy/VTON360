import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPImageProcessor
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

# https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train_plus.py#L49

class ReferenceEncoder(nn.Module):
    def __init__(self, model_path="openai/clip-vit-base-patch32"):
        super(ReferenceEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path,local_files_only=True)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
        
        # pooled_output = outputs.pooler_output
        # return pooled_output




class ReferenceEncoder2(nn.Module):
    def __init__(self, model_path="openai/clip-vit-base-patch32"):
        super(ReferenceEncoder2, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path,local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(model_path,local_files_only=True)
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        
        print(inputs['pixel_values'].size())
        
        outputs = self.model(**inputs)
        print(outputs['last_hidden_state'].shape)
        print(outputs.keys())
        pooled_output = outputs.pooler_output

        return pooled_output

# # example
# model = ReferenceEncoder2(model_path='/root/autodl-tmp/Open-AnimateAnyone/pretrained_models/clip-vit-base-patch32')
# image_path = "../test.png"
# # image_path = "/mnt/f/research/HumanVideo/AnimateAnyone-unofficial/DWPose/0001.png"
# image = Image.open(image_path).convert('RGB')
# image = [image,image]

# pooled_output = model(image)

# print(f"Pooled Output Size: {pooled_output.size()}") # Pooled Output Size: torch.Size([bs, 768])
