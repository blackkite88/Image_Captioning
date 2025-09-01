import os
from pickle import dump
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.text_processing import cleaning_text, all_img_captions, text_vocabulary, save_descriptions
from utils.data_loader import load_photos, load_clean_descriptions, load_features
from utils.sequence_utils import dict_to_list, data_generator
from model.caption_model import define_model

# paths
dataset_text = "data/Flickr8k_text"
dataset_images = "data/Flicker8k_Dataset"

# prepare data
filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename, dataset_images)
train_descriptions = load_clean_descriptions("data/descriptions.txt", train_imgs)
train_features = load_features("data/features.p", train_imgs)

# tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dict_to_list(train_descriptions))
dump(tokenizer, open("tokenizer.p", "wb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(d.split()) for d in dict_to_list(train_descriptions))

# model
model = define_model(vocab_size, max_length)

# training
epochs = 10
steps = sum(len(c.split()) - 1 for caps in train_descriptions.values() for c in caps) // 32
os.makedirs("models2", exist_ok=True)

for i in range(epochs):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    model.fit(dataset, epochs=4, steps_per_epoch=steps, verbose=1)
    model.save(f"models2/model_{i}.h5")
