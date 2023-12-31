from typing import Type, List
import sys

import numpy as np

from similarity import ast
from similarity.config import LanguageConfig
from similarity.vocabulary import Vocabulary


# 将AST树转变为语义矩阵
class ASTTransformer:
    def __init__(self, lang, vocabulary, vocabulary_offset=0, input_length=None):
        self.language = lang
        self.vocabulary = vocabulary # 词法单元表
        self.vocabulary_offset = np.int32(vocabulary_offset)
        self.input_length = input_length
        self.total_input_length = input_length

    def transform_ast(self, list_ast):
        raise NotImplementedError()

    def nodes_to_indexes(self, nodes):
        indexes = [self.node_index(node) for node in nodes]
        if not self.input_length:
            return indexes
        if len(indexes) > self.input_length:
            return False
        return self.pad(indexes)

    def node_index(self, node):
        return self.vocabulary.index(node) + self.vocabulary_offset

    def pad(self, indexes, pad_value=np.int32(0)):
        return indexes + [pad_value] * (self.input_length - len(indexes))

# 深度优先遍历填写矩阵的列顺序
class DFSTransformer(ASTTransformer):
    def transform_ast(self, list_ast):
        return self.nodes_to_indexes(list_ast)

# 宽度优先遍历填写矩阵的列顺序
class BFSTransformer(ASTTransformer):
    def transform_ast(self, list_ast):
        ast_root = ast.from_list(list_ast)
        return self.nodes_to_indexes(ast_root.bfs())


class MultiTransformer(ASTTransformer):
    def __init__(self, lang, vocabulary, vocabulary_offset=0, input_length=None):
        super(MultiTransformer, self).__init__(lang, vocabulary, vocabulary_offset, input_length)
        if self.total_input_length:
            self.total_input_length *= 2

    @property
    def split_input(self):
        return True


class DBFSTransformer(MultiTransformer):
    def transform_ast(self, list_ast):
        ast_root = ast.from_list(list_ast)
        return self.nodes_to_indexes(ast_root.dfs()) + \
               self.nodes_to_indexes(ast_root.bfs())


class BiDFSTransformer(MultiTransformer):
    def transform_ast(self, list_ast):
        ast_root = ast.from_list(list_ast)
        return self.nodes_to_indexes(ast_root.dfs()) + \
               self.nodes_to_indexes(ast_root.dfs(reverse=True))


def get_class(language_config: LanguageConfig) -> Type[ASTTransformer]:
    return getattr(thismodule, language_config.transformer_class_name)


def create_all(languages: List[LanguageConfig]) -> List[ASTTransformer]:
    return [create(lang) for lang in languages]


