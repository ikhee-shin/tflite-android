import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import re
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


class Tokenizer:
    def __init__(self, vocab_path, max_seq_len=64):

        self._max_seq_len = max_seq_len

        self._PAD = "<PAD>"
        self._UNK = "<UNKNOWN>"
        self._START = "<START>"

        self._tokenToIdx = {}
        self._vocab_size = 0

        self.load_vocab(vocab_path)

    def load_vocab(self, vocab_path):

        with open(vocab_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                try:
                    token, idx = line.split()
                except:
                    print(line)
                self._tokenToIdx[token] = int(idx)

        # exception
        self._tokenToIdx[" "] = 5138

        self._vocab_size = len(self._tokenToIdx)

    def tokenize(self, utterance):

        indices = []

        utterance = utterance.strip()
        # words = utterance.split()
        # https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
        words = re.split(' |; |, |\\. |! |\\? |\\*|\n', utterance)

        if self._START in self._tokenToIdx.keys():
            indices.append(self._tokenToIdx[self._START])

        for word in words:
            if word in self._tokenToIdx.keys():
                indices.append(self._tokenToIdx[word])
            else:
                indices.append(self._tokenToIdx[self._UNK])

        indices = indices[:self._max_seq_len]
        indices += [self._tokenToIdx[self._PAD]] * max(0, self._max_seq_len - len(indices))

        if len(indices) != self._max_seq_len:
            raise Exception("length of indices is not equal to max seq len")

        return indices


def main():
    # Split the training set into 60% and 40%, so we'll end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.

    vocab_path = "./text_classification_vocab.txt"
    max_seq_len = 256
    tokenizer = Tokenizer(vocab_path, max_seq_len)

    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)

    train_utterance = [data[0].numpy().decode("utf-8").strip() for data in train_data]
    train_labels = np.array([data[1].numpy() for data in train_data])

    test_utterance = [data[0].numpy().decode("utf-8").strip() for data in test_data]
    test_labels = np.array([data[1].numpy() for data in test_data])

    train_indices = np.array([tokenizer.tokenize(utterance) for utterance in train_utterance])
    test_indices = np.array([tokenizer.tokenize(utterance) for utterance in test_utterance])

    print(train_indices[0])

    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(max_seq_len,), name="input"),
        keras.layers.Embedding(
            tokenizer._vocab_size, 32,
            embeddings_constraint=None, mask_zero=False, input_length=max_seq_len
        ),
        # keras.layers.LayerNormalization(),
        keras.layers.Dense(16, activation="relu", name="dense1"),
        # keras.layers.LayerNormalization(),
        keras.layers.Dense(4, activation="relu", name="dense2"),
        keras.layers.Flatten(input_shape=(max_seq_len, 4), name="flatten"),
        keras.layers.Dense(256, activation="relu", name="dense3"),
        # keras.layers.LayerNormalization(),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(128, activation="relu", name="dense4"),
        # keras.layers.LayerNormalization(),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(64, activation="relu", name="dense5"),
        # keras.layers.LayerNormalization(),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(2, activation="softmax", name="output")
    ], name="text_classification")

    model.summary()
    # model.compile(optimizer="adam",
    #               loss=keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=["accuracy"])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x={"input": train_indices}, y={"output": train_labels}, epochs=20)

    # Test model
    test_loss, test_acc = model.evaluate(x={"input": test_indices},
                                         y={"output": test_labels},
                                         verbose=2)

    # save as tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    f = open('text_clf.tflite', 'wb')
    f.write(tflite_model)
    f.close()

    print("-" * 50)
    print("Test accuracy: ")
    print(test_acc)

    tf.saved_model.save(model, "./models/simple_model")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="simple_frozen_graph.pb",
                      as_text=False)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/simple_frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)


if __name__ == "__main__":
    main()