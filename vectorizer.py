from typing import List

import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import h5py
from tqdm import tqdm

from similarity.file_processor import FileProcessor
from similarity.config import Config
from similarity.layers import ModelWrapper
from similarity import util


# 将语义矩阵转换为向量,用于后续LSTM模型学习
class Vectorizer(FileProcessor):
    def __init__(self, config: Config, model: ModelWrapper, options: dict = None):
        super(Vectorizer, self).__init__(config, model, options)
        self.encoders = {}
        for i, lang in enumerate(config.model.languages):
            encoder_index = options.get("encoder_index")
            if not lang.name in self.encoders and \
                (not encoder_index or encoder_index == i):
                self.encoders[lang.name] = model.inner_models[i]

    def vectorize(self, filenames: List[str], language: str, sess: tf.Session):
        input_filenames = []
        input_vectors = []
        for filename in filenames:
            _ast, input_vector = self.get_file_vector(filename, language)
            if input_vector:
                input_filenames.append(filename)
                input_vectors.append(input_vector)
        model_input = tf.constant(pad_sequences(input_vectors))
        vectors = sess.run(self.encoders[language](model_input))
        return zip(input_filenames, vectors)

    def process(self, input_filenames: List[str], output: str):
        by_lang = self._group_filenames(input_filenames)
        batch_size = self.options.get("batch_size") or self.config.trainer.batch_size
        sess = K.get_session()
        with h5py.File(output, "w") as f, \
             tqdm(total=len(input_filenames)) as pbar:
            for lang, lang_filenames in by_lang.items():
                for filenames in util.in_batch(lang_filenames, batch_size):
                    for filename, vector in self.vectorize(filenames, lang, sess):
                        f.create_dataset(filename, data=vector)
                    pbar.update(len(filenames))

    def _group_filenames(self, filenames):
        key = lambda filename: util.filename_language(filename, self.encoders)
        return util.group_by(filenames, key)
