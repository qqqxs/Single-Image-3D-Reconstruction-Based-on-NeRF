import torch
import clip
from PIL import Image


def get_clip_score(image_path1, image_path2):
    # Load the pre-trained CLIP model and the image
    model, preprocess = clip.load('ViT-B/32')
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Preprocess the image and tokenize the text
    image1_input = preprocess(image1).unsqueeze(0)
    image2_input = preprocess(image2).unsqueeze(0)

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image1_input = image1_input.to(device)
    image2_input = image2_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image1_features = model.encode_image(image1_input)
        image2_features = model.encode_image(image2_input)

    # Normalize the features
    image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = (image1_features * image2_features).sum(-1).mean()

    return clip_score


image1_path = "nv imgs/dragon/nv-coarse.png"
image2_path = "nv imgs/dragon/nv-refine.png"
image3_path = "nv imgs/dragon/df.png"
image4_path = "nv imgs/dragon/pe.png"
# image5_path = "nv imgs/dragon/ref-coarse.png"
# image6_path = "nv imgs/dragon/ref-refine.png"
image7_path = "ref imgs/dragon/dragon.png"
# image8_path = "nv imgs/dragon/ref-pe.png"

coarse_score = get_clip_score(image1_path, image7_path)
refine_score = get_clip_score(image2_path, image7_path)
pe_score = get_clip_score(image3_path, image7_path)
df_score = get_clip_score(image4_path, image7_path)
print(f"Coarse CLIP Score: {coarse_score}")
print(f"Refine CLIP Score: {refine_score}")
print(f"Point-E CLIP Score: {pe_score}")
print(f"DreamFusion CLIP Score: {df_score}")
