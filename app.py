import os
import pathlib
import pickle
import zipfile

import gradio as gr
from PIL import Image
from sentence_transformers import util
from tqdm.autonotebook import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

img_folder = "photos/"
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)

    photo_filename = "unsplash-25k-photos.zip"
    if not os.path.exists(
        photo_filename
    ):  # Download dataset if does not exist
        util.http_get(
            "http://sbert.net/datasets/" + photo_filename, photo_filename
        )

    # Extract all images
    with zipfile.ZipFile(photo_filename, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            zf.extract(member, img_folder)

cwd = pathlib.Path(__file__).parent.absolute()

# Define model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

emb_filename = "unsplash-25k-photos-embeddings.pkl"
emb_path = cwd / emb_filename
if not os.path.exists(emb_filename):
    util.http_get("http://sbert.net/datasets/" + emb_filename, emb_path)
with open(emb_path, "rb") as fIn:
    img_names, img_emb = pickle.load(fIn)


def search_text(query):
    """Search an image based on the text query.

    Args:
        query ([string]): [query you want search for]
    Returns:
        [list]: [list of images that are related to the query.]
    """
    inputs = tokenizer([query], padding=True, return_tensors="pt")
    query_emb = model.get_text_features(**inputs)
    hits = util.semantic_search(query_emb, img_emb, top_k=8)[0]
    images = [
        (
            Image.open(cwd / "photos" / img_names[hit["corpus_id"]]),
            f"{hit['score']*100:.2f}%",
        )
        for hit in hits
    ]
    return images


gr.Interface(
    title="Text2Image search using CLIP model 🔤 ➡️  📸",
    description="This is a demo of the CLIP model. We use the CLIP model to search for images from the unsplash dataset based on a text query.",
    article="By Julien Assuied, Elie Brosset, Lucas Chapron et Alexis Japas",
    fn=search_text,
    theme="gradio/soft",
    allow_flagging="never",
    inputs=[
        gr.Textbox(
            lines=2,
            label="What do you want to see ?",
            placeholder="Write the prompt here...",
        ),
    ],
    outputs=[
        gr.Gallery(
            label="Most similar images", show_label=True, elem_id="gallery"
        ).style(grid=[2], height="auto"),
    ],
    examples=[
        [("Two cats")],
        [("A plane flying")],
        [("A family picture")],
        [("un homme marchant sur le parc")],
    ],
).launch(debug=True)
