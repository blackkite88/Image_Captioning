import os
from pickle import load
from .text_processing import load_doc

def load_photos(filename, dataset_images):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return [p for p in photos if os.path.exists(os.path.join(dataset_images, p))]

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions.setdefault(image, []).append(desc)
    return descriptions

def load_features(filename, photos):
    all_features = load(open(filename, "rb"))
    return {k: all_features[k] for k in photos}
