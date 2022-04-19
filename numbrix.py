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
from tkinter.messagebox import NO

from more_itertools import adjacent
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search
from utils import unique
from itertools import chain, combinations  # TODO podemos importar isto? (é usado no utils)


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        s_size = self.longest_sequence_size()
        o_size = other.longest_sequence_size()
        if s_size != o_size:
            return s_size > o_size

        s_size = self.largest_free_area()
        o_size = other.largest_free_area()
        if s_size != o_size:
            return s_size > o_size

        return self.id < other.id

    def recursive_sequence_counter(self, explored, row, col, val):
        """ Devolve o comprimento da sequência de números seguidos que contem um 
        dado valor numa dada posicao """
        ret = 1
        adjacents = self.board.get_all_adjacents(row, col)

        # encontra o valor seguinte na cadeia
        if val + 1 in adjacents and val + 1 not in explored:
            new_row, new_col = self.board.find_number(val + 1)
            explored += [val + 1]
            ret += self.recursive_sequence_counter(explored, new_row, new_col, val + 1)

        # encontra o valor anterior na cadeia
        if val - 1 in adjacents and val - 1 not in explored and val - 1 != 0:
            new_row, new_col = self.board.find_number(val - 1)
            explored += [val - 1]
            ret += self.recursive_sequence_counter(explored, new_row, new_col, val - 1)

        return ret

    def longest_sequence_size(self) -> int:
        """ Devolve o comprimento da sequência de números seguidos mais longa 
        no tabuleiro """
        max = 0
        explored = []
        for row in range(self.board.N):
            for col in range(self.board.N):
                val = self.board.get_number(row, col)
                if val != 0 and val not in explored:
                    explored += [val]
                    ret = self.recursive_sequence_counter(explored, row, col, val)
                    if ret > max:
                        max = ret
                if max > self.board.N / 2:
                    return max
        return max

    def recursive_free_area_counter(self, explored, row, col):
        ret = 1
        adjacents = self.board.get_all_adjacents(row, col)

        #abaixo
        if adjacents[0] == 0 and (row + 1, col) not in explored:
            explored += [(row + 1, col)]
            ret += self.recursive_free_area_counter(explored, row + 1, col)
            
        #acima
        if adjacents[1] == 0 and (row - 1, col) not in explored:
            explored += [(row - 1, col)]
            ret += self.recursive_free_area_counter(explored, row - 1, col)
        
        #esquerda
        if adjacents[2] == 0 and (row, col - 1) not in explored:
            explored += [(row, col - 1)]
            ret += self.recursive_free_area_counter(explored, row, col - 1)
        
        #direita
        if adjacents[3] == 0 and (row, col + 1) not in explored:
            explored += [(row, col + 1)]
            ret += self.recursive_free_area_counter(explored, row, col + 1)

        return ret

    def largest_free_area(self) -> int:
        max = 0
        explored = []
        for row in range(self.board.N):
            for col in range(self.board.N):
                val = self.board.get_number(row, col)
                if val == 0 and (row, col) not in explored:
                    explored += [(row, col)]
                    ret = self.recursive_free_area_counter(explored, row, col)
                    if ret > max:
                        max = ret
                if max > self.board.N / 2:
                    return max
        return max


    def recursive_path_counter(self, path, len, obj_len, obj, found):
        """ Devolve o comprimento da sequência de números seguidos que contem um 
        dado valor numa dada posicao """
        row, col = path[-1]
        adjacents = self.board.get_all_adjacents(row, col)

        if len == obj_len and obj in adjacents:
            found = True
            return found

        #abaixo
        if adjacents[0] == 0 and (row + 1, col) not in path and not found:
            found = self.recursive_path_counter(path + [(row + 1, col)], len + 1, obj_len, obj, found)
            
        #acima
        if adjacents[1] == 0 and (row - 1, col) not in path and not found:
            found = self.recursive_path_counter(path + [(row - 1, col)], len + 1, obj_len, obj, found)
        
        #esquerda
        if adjacents[2] == 0 and (row, col - 1) not in path and not found:
            found = self.recursive_path_counter(path + [(row, col - 1)], len + 1, obj_len, obj, found)
        
        #direita
        if adjacents[3] == 0 and (row, col + 1) not in path and not found:
            found = self.recursive_path_counter(path + [(row, col + 1)], len + 1, obj_len, obj, found)

        return found
    
    def exists_valid_path_between(self, val1, val2):
        row, col = self.board.find_number(val1)
        obj_len = val2 - val1 - 1
        if self.recursive_path_counter([(row, col)], 0, obj_len, val2, False):
            return True
        return False

    # TODO: outros metodos da classe


class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    def __init__(self, N: int, board: list):
        self.N = N
        self.board = board

    def get_number(self, row: int, col: int) -> int | None:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        if 0 <= row < self.N and 0 <= col < self.N:
            return self.board[row][col]

        return None

    def set_number(self, row: int, col: int, value: int):
        """ Coloca o valor na respetiva posição do tabuleiro. """
        self.board[row][col] = value

    def find_number(self, value: int) -> tuple[int, int] | tuple[None, None]:
        """ Devolve a localização do valor no tabuleiro. """
        for rows in range(self.N):
            for cols in range(self.N):
                if self.board[rows][cols] == value:
                    return rows, cols

        return None, None

    def placed_values(self) -> set:
        """ Devolve a lista de valores que foram colocados no tabuleiro """
        return sorted(set(v for row in self.board for v in row if v != 0))

    def possible_values(self) -> set:
        """ Devolve a lista de valores que não foram ainda colocados
        no tabuleiro """
        all_values = set(range(1, self.N ** 2 + 1))
        return all_values.difference(self.placed_values())

    def find_mininum(self) -> tuple[int | None, int | None, int]:
        # TODO descricao e ver onde se usa
        minimum = self.N ** 2 + 1
        row = None
        col = None

        for rows in range(self.N):
            for cols in range(self.N):
                value = self.board[rows][cols]
                if value < minimum:
                    row = rows
                    col = cols
                    minimum = value

        return row, col, minimum

    def find_maximum(self) -> tuple[int | None, int | None, int]:
        # TODO descricao e ver onde se usa
        maximum = 0
        row = None
        col = None

        for rows in range(self.N):
            for cols in range(self.N):
                value = self.board[rows][cols]
                if value > maximum:
                    row = rows
                    col = cols
                    maximum = value

        return row, col, maximum

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

    def get_copy(self):
        """ Devolve uma cópia da Board. """
        copy_board = [line[:] for line in self.board]
        return Board(self.N, copy_board)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        with open(filename, 'r') as file:
            N = int(file.readline()[:-1])
            board = list(map(lambda x: list(map(int, x[:-1].split('\t'))), file.readlines()))

        return Board(N, board)

    def __repr__(self):
        ret = ""

        for line in self.board:
            ret += '\t'.join(map(str, line))
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
        
    def is_valid_action(self, state: NumbrixState, action: tuple[int, int, int]) -> bool:
        """ Retorna um booleano referente à possibilidade de executar a 'action'
        passada como argumento sobre o 'state' passado como argumento.
        Verifica se a posicao da 'action' pertence ao 'board' e se o valor
        aplicado é plausível. """

        # TODO é possível por isto a cortar jogadas q nao vao fazer sentido no futuro
        # mas dá um bcd trabalho e se calhar a heuristica faz isto / é suficiente

        adjacents = state.board.get_all_adjacents(action[0], action[1])
        possible_values = state.board.possible_values()

        # o valor não está a ser colocado ao lado do seu sucessor/antecessor que já está no tabuleiro
        if ((action[2] != state.board.N ** 2 and action[2] + 1 not in adjacents and action[
            2] + 1 not in possible_values)
                or (action[2] != 1 and action[2] - 1 not in adjacents and action[2] - 1 not in possible_values)):
            return False

        # o valor está ser colocado impossibilitando a colocação futura do seu sucessor/antecessor
        if (adjacents.count(0) == 0
                and (action[2] + 1 in possible_values or action[2] - 1 in possible_values)):
            return False
        # TODO ver pq nao entra neste if
        if (adjacents.count(0) == 1
                and action[2] + 1 in possible_values and action[2] - 1 in possible_values):
            return False

        new_state = self.result(state, action)
        possible_values = new_state.board.possible_values()

        # o valor está ser colocado impossibilitando a colocação futura do sucessor/antecessor
        # de um dos seus números adjacentes
        for val in adjacents:
            if val is not None:
                row, col = new_state.board.find_number(val)
                val_adjacents = new_state.board.get_all_adjacents(row, col)
                if (val_adjacents.count(0) == 0
                        and (val + 1 in possible_values or val - 1 in possible_values)):
                    return False
                if (val_adjacents.count(0) == 1
                        and val + 1 in possible_values and val - 1 in possible_values):
                    return False

        # o valor esta a ser colocado de forma a ficarem espacos vazios 
        # entre o numero x-1 e x+1 sabendo que x já foi colocado noutro sitio
        for row in range(new_state.board.N):
            for col in range(new_state.board.N):
                val = new_state.board.get_number(row, col)
                if val == 0:
                    adjacents = new_state.board.get_all_adjacents(row, col)
                    if adjacents.count(0) == 0:
                        pairs = list(chain.from_iterable(combinations(adjacents, r) for r in range(2, 3)))[1:]
                        for pair in pairs:
                            if pair[1] is None or pair[0] is None:
                                continue
                            if pair[1] - pair[0] == 2 and pair[1] - 1 not in possible_values:
                                return False
        
        # o valor colocado impede a ligação entre os valores colocados
        placed_values = new_state.board.placed_values()
        for i in range(len(placed_values) - 1):
            if not new_state.exists_valid_path_between(placed_values[i], placed_values[i + 1]):
                return False

        return True

    def actions(self, state: NumbrixState) -> list[tuple[int, int, int]]:
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        # FIXME se não for usado apagar funções
        # max_row, max_col, maximum = state.board.find_maximum()
        # min_row, min_col, minimum = state.board.find_mininum()
        possible_vals = state.board.possible_values()
        ret = []

        # percorrer todas as posições vazias do tabuleiro
        for row in range(state.board.N):
            for col in range(state.board.N):
                value = state.board.get_number(row, col)
                if value == 0:
                    adjacents = state.board.get_all_adjacents(row, col)

                    # todas as posições adjacentes estão preenchidas, logo ação obrigatória
                    if adjacents.count(0) == 0:
                        ret = []
                        pairs = list(chain.from_iterable(combinations(adjacents, r) for r in range(2, 3)))[1:]

                        # descobrir qual o valor a colocar entre os adjacentes val+1 e val-1
                        for pair in pairs:
                            if pair[1] is None or pair[0] is None:
                                continue
                            if pair[1] - pair[0] == 2 and self.is_valid_action(state, (row, col, pair[1] - 1)):
                                return [(row, col, pair[1] - 1)]
                        if self.is_valid_action(state, (row, col, 1)):
                            return [(row, col, 1)]
                        elif self.is_valid_action(state, (row, col, state.board.N ** 2)):
                            return [(row, col, state.board.N ** 2)]

                    # adicionar à lista de ações colocar o antecessor/sucessor de um valor
                    # nas posições adjacentes se for possível
                    else:
                        test = [i + 1 for i in adjacents if i is not None and i != 0 and i != state.board.N ** 2]
                        test += [i - 1 for i in adjacents if i is not None and i != 0 and i != 1]
                        for val in test:
                            if val in possible_vals and self.is_valid_action(state, (row, col, val)):
                                ret += [(row, col, val)]
        return unique(ret)

    def result(self, state: NumbrixState, action) -> NumbrixState:
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
        i = 1
        row, col = state.board.find_number(i)

        if row is None or col is None:
            return False

        # Procura pelas restantes posições
        while i < state.board.N ** 2:
            i += 1
            horizontal = state.board.adjacent_horizontal_numbers(row, col)
            vertical = state.board.adjacent_vertical_numbers(row, col)
            if horizontal[0] == i:
                col -= 1
            elif horizontal[1] == i:
                col += 1
            elif vertical[0] == i:
                row += 1
            elif vertical[1] == i:
                row -= 1
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
            if action[2] + 1 in action_adjacents and action[2] - 1 in action_adjacents:
                return 0
            if (action[2] + 1 in action_adjacents or action[2] - 1 in action_adjacents) \
                    and action_adjacents.count(0) == 0:
                return 0
        # TODO reduce complexity
        return self.initial.board.N ** 2 - state.longest_sequence_size() - state.largest_free_area()

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