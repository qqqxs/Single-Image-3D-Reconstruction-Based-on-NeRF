from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)
imgfile = 'demo/niuzimao.png'
image = Image.open(imgfile).convert('RGB')

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# an astronaut in a white spacesuit standing on a hill
# a model of a house on a green field
# a dog is running in the air on a black background
# a brown teddy bear sitting on a black background
# a small rabbit is sitting on top of a stack of pancakes
# a blue bird sitting on a wicker basket of macarons
# back to the future delorean car
# a stuffed toy bird with a blue and white striped coat
# a stuffed animal with brown fur and white eyes     dtu1
# smurfs plush toy with a red hat and blue pants     dtu2
# a single orange pumpkin on a black background      dtu3
# a stuffed animal with brown fur and white eyes     dtu4
# a skull with a missing tooth on a black background dtu5
# a stuffed pig is shown on a black background       dtu6
# a statue of a rabbit sitting on a black background dtu7
# a bottle of coffee on a black background           dtu8
# a model of a house with a red roof                 dtu9
# a model of a building with a red roof              dtu10
# a young man in a suit and tie
# a cat is laying down with its mouth open
