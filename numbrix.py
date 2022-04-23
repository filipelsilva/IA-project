# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 10:
# 95533 André Martins Esgalhado
# 95574 Filipe Ligeiro Silva

# FIXME isto pode falhar no mooshak, depois teremos que testar, devido às type
# annotations. Caso falhe, podemos apenas voltar a colocar como estava, o
# código corre na mesma
from __future__ import annotations

import sys
from unittest import result

from more_itertools import adjacent
from sympy import re
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search
from utils import unique
from itertools import chain, combinations  # TODO podemos importar isto? (é usado no utils)
import math

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        if len(self.board.board) != len(other.board.board):
            return len(self.board.board) > len(other.board.board)
        return self.id < other.id
        
    # TODO: outros metodos da classe


class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    def __init__(self, N: int, board: dict):
        self.N = N
        self.board = board
    
    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        if row >= self.N or row < 0 or col >= self.N or col < 0:
            return None
        if (row, col) in self.board:
            return self.board[(row, col)]
        return 0
    
    def set_number(self, row: int, col: int, value: int):
        """ Coloca o valor na respetiva posição do tabuleiro. """
        self.board[(row,col)] = value

    def find_number(self, value: int) -> tuple[int, int] | tuple[None, None]:
        """ Devolve a localização do valor no tabuleiro. """
        for place in self.board:
            if self.board[place] == value:
                return place

        return None, None
    
    def get_placed_values(self) -> set:
        """ Devolve a lista de valores que foram colocados no tabuleiro """
        return self.board.values()

    def get_possible_values(self) -> set:
        """ Devolve a lista de valores que não foram ainda colocados
        no tabuleiro """
        all_values = set(range(1, self.N ** 2 + 1))
        return all_values.difference(self.board.values())
    
    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple[int | None, int | None]:
        """ Devolve os valores imediatamente abaixo e acima,
        respectivamente. """
        return self.get_number(row + 1, col), self.get_number(row - 1, col)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple[int | None, int | None]:
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        return self.get_number(row, col - 1), self.get_number(row, col + 1)

    def get_all_adjacents(self, row, col):
        """ Devolve os valores imediatamente abaixo, acima, à esquerda e 
        à direita, respetivamente. """
        adjacents = self.adjacent_vertical_numbers(row, col)
        adjacents += self.adjacent_horizontal_numbers(row, col)
        return adjacents

    def get_free_adjacent_positions(self, row, col):
        ret = []
        if row + 1 < self.N  and self.get_number(row + 1, col) == 0:
            ret += [(row + 1, col)]
        if row > 0 and self.get_number(row - 1, col) == 0:
            ret += [(row - 1, col)]
        if col + 1 < self.N and self.get_number(row, col + 1) == 0:
            ret += [(row, col + 1)]
        if col > 0 and self.get_number(row, col - 1) == 0:
            ret += [(row, col - 1)]
        return ret
    
    def get_copy(self):
        """ Devolve uma cópia da Board. """
        copy_board = {}
        for place in self.board:
            copy_board[place] = self.board[place]
        return Board(self.N, copy_board)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        with open(filename, 'r') as file:
            N = int(file.readline()[:-1])
            board_tmp = list(map(lambda x: list(map(int, x[:-1].split('\t'))), file.readlines()))
            board = dict()
            for row in range(N):
                for col in range(N):
                    if board_tmp[row][col] != 0:
                        board[(row, col)] = board_tmp[row][col]

        return Board(N, board)

    def __repr__(self):
        ret = ""

        for row in range(self.N):
            for col in range(self.N):
                if (row, col) in self.board:
                    ret += str(self.board[(row, col)])
                else:
                    ret += '0'
                ret += '\t'
            ret += '\n'

        return ret

    def to_string(self) -> str:
        """ Retorna representação em string da Board. É igual a __repr__, para
        compatibilidade com os exemplos do enunciado. """
        return self.__repr__()

    # TODO: outros metodos da classe


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)
        self.N = 0

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        ret = []

        board = dict(sorted(state.board.board.items(), key=lambda item: item[1]))
        possible_values = state.board.get_possible_values()

        for place in board:
            row, col = place
            value = state.board.get_number(row, col)
            free_positions = state.board.get_free_adjacent_positions(row, col)
                
            for position in free_positions:
                r, c = position
                adjacents = state.board.get_all_adjacents(r, c)
                for v in adjacents:
                    if v is not None and v != 0 and v != value:
                        if abs(v - value) == 2:
                            if max(v, value) - 1 in possible_values:
                                ret += [(r, c, max(v, value) - 1)]
            if len(ret) > 0:
                return ret

        for place in board:
            row, col = place
            value = state.board.get_number(row, col)
            free_positions = state.board.get_free_adjacent_positions(row, col)
                
            for position in free_positions:
                if value + 1 <= state.board.N ** 2 and value + 1 in possible_values:
                    ret += [(position[0], position[1], value + 1)]
                if value - 1 > 0 and value - 1 in possible_values:
                    ret += [(position[0], position[1], value - 1)]
            if len(ret) > 0:
                return ret

        return ret


    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de 
        self.actions(state). """
        copy_board = state.board.get_copy()
        row, col, value = action
        copy_board.set_number(row, col, value)

        return NumbrixState(copy_board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. """
        self.N += 1
        print(state.board)
        if len(state.board.board) != state.board.N ** 2:
            return False

        i = 1
        row, col = state.board.find_number(i)

        if row is None or col is None:
            return False

        # Procura pelas restantes posições
        while i < state.board.N ** 2:
            i += 1
            adjacents = state.board.get_all_adjacents(row, col)
            if adjacents[0] == i:
                row += 1
            elif adjacents[1] == i:
                row -= 1
            elif adjacents[2] == i:
                col -= 1
            elif adjacents[3] == i:
                col += 1
            else:
                return False
        
        return True

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        action = node.action
        state = node.state
        if action is not None:
            action_adjacents = state.board.get_all_adjacents(action[0], action[1])
            # best option
            if action[2] != 1 and action[2] + 1 in action_adjacents and action[2] - 1 in action_adjacents:
                return -math.inf
            if action_adjacents.count(0) == 0:
                if action[2] == 1 and action[2] + 1 in action_adjacents:
                    return -math.inf
                if action[2] == state.board.N ** 2 in action_adjacents:
                    return -math.inf
        return math.inf
    
    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    # if len(sys.argv) != 2:
    #     print("Usage: python3 numbrix.py <instance_file>")
    #     sys.exit(1)

    board = Board.parse_instance(sys.argv[1])

    # i1.txt do enunciado
    # board = Board(3, [[0,0,0],[0,0,2],[0,6,0]])
    # board = Board(3, [[9,4,3],[8,5,2],[7,6,1]])

    problem = Numbrix(board)
    goal_node = greedy_search(problem)
    print("Is goal?", problem.goal_test(goal_node.state))
    print("Solution:\n", goal_node.state.board.to_string(), sep="")
    print(problem.N)