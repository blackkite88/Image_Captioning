import string

def load_doc(filename):
    with open(filename, 'r') as file:
        return file.read()

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        descriptions.setdefault(img[:-2], []).append(caption)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            desc = img_caption.replace("-", " ").split()
            desc = [w.lower() for w in desc]
            desc = [w.translate(table) for w in desc]
            desc = [w for w in desc if len(w) > 1 and w.isalpha()]
            captions[img][i] = ' '.join(desc)
    return captions

def text_vocabulary(descriptions):
    vocab = set()
    for descs in descriptions.values():
        [vocab.update(d.split()) for d in descs]
    return vocab

def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    with open(filename, "w") as file:
        file.write("\n".join(lines))
