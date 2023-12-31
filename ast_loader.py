import random
import logging
import gzip
import json
from os import path


# 加载Anltr4生成的AST树
class ASTLoader:
    def __init__(self, asts_path, filenames_path=None, file_type="multi_file"):
        if filenames_path is None:
            filenames_path = path.splitext(asts_path)[0] + ".txt"
        if file_type == "multi_file" and not path.exists(filenames_path):
            logging.warning("%s不存在", filenames_path)
            file_type = "single_file"
        if file_type == "multi_file":
            if filenames_path is None:
                filenames_path = path.splitext(asts_path)[0] + ".txt"
            self._load_asts(asts_path)
            self._load_names(filenames_path)
        elif file_type == "single_file":
            self._load_single_file_format(asts_path)

    def _load_single_file_format(self, filepath):
        with self._open(filepath) as f:
            entries = [json.loads(row) for row in f]
            self.names = {entry["filename"]: index for (index, entry) in enumerate(entries)}
            self.asts = [entry["tokens"] for entry in entries]

    def _load_names(self, names_path):
        with self._open(names_path) as f:
            self.names = {filename.strip(): index for (index, filename) in enumerate(f)}

    def _load_asts(self, asts_path):
        with self._open(asts_path) as f:
            self.asts = [json.loads(ast) for ast in f]

    def get_ast(self, filename):
        return self.asts[self.names[filename]]

    def random_ast(self, predicate=tautology):
        keys = list(self.names.keys())
        while True:
            name = random.choice(keys)
            ast = self.get_ast(name)
            if predicate(name, ast):
                return name, ast

    def has_file(self, filename):
        return filename in self.names

    def open(filename):
        if filename.endswith(".gz"):
            return gzip.open(filename)
        else:
            return open(filename)
