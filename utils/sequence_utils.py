import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def dict_to_list(descriptions):
    all_desc = []
    for descs in descriptions.values():
        all_desc.extend(descs)
    return all_desc

def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    def generator():
        while True:
            for key, desc_list in descriptions.items():
                feature = features[key][0]
                X1, X2, y = create_sequences(tokenizer, max_length, desc_list, feature, vocab_size)
                for i in range(len(X1)):
                    yield {'input_1': X1[i], 'input_2': X2[i]}, y[i]

    output_signature = (
        {'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
         'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)},
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(32)
