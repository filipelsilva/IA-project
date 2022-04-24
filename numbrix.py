# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 10:
# 95533 André Martins Esgalhado
# 95574 Filipe Ligeiro Silva

import sys

from numpy import place

from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search
import math
import bisect


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other) -> bool:
        s_areas = self.board.get_free_areas()
        o_areas = other.board.get_free_areas()
        if s_areas != o_areas:
            return s_areas < o_areas
        if len(self.board.board) != len(other.board.board):
            return len(self.board.board) > len(other.board.board)

        return self.id < other.id

    def impossible_free_area(self, free_area) -> bool:
        frontier = self.board.get_frontier(free_area)
        max_len = len(free_area)

        for val in frontier:
            r, c = self.board.find_number(val)
            adjacents = self.board.get_all_adjacents(r, c)
            if self.val_in_place(val, adjacents):
                del frontier[frontier.index(val)]
                continue
            for other in frontier:
                if val == other:
                    continue
                r, c = self.board.find_number(other)
                adjacents = self.board.get_all_adjacents(r, c)
                if self.val_in_place(other, adjacents):
                    del frontier[frontier.index(other)]
                    continue
                if max_len < abs(val - other) - 1:
                    del frontier[frontier.index(val)]
                    del frontier[frontier.index(other)]
                    break
        if len(frontier) != 0:
            return False
        return True

    def has_impossible_free_areas(self):
        for area in self.board.get_free_areas():
            if self.impossible_free_area(area):
                return True
        return False
            

    def recursive_path_counter(self, path, length, obj_len, obj, found) -> bool:
        """ Função auxiliar para verificar se existe caminho válido entre dois valores da Board. """
        row, col = path[-1]
        adjacents = self.board.get_all_adjacents(row, col)

        if length == obj_len:
            if obj in adjacents or obj == 1 or obj == self.board.N ** 2:
                return True

            return False

        # Abaixo
        if adjacents[0] == 0 and (row + 1, col) not in path and not found:
            found = self.recursive_path_counter(path + [(row + 1, col)], length + 1, obj_len, obj, found)

        # Acima
        if adjacents[1] == 0 and (row - 1, col) not in path and not found:
            found = self.recursive_path_counter(path + [(row - 1, col)], length + 1, obj_len, obj, found)

        # Esquerda
        if adjacents[2] == 0 and (row, col - 1) not in path and not found:
            found = self.recursive_path_counter(path + [(row, col - 1)], length + 1, obj_len, obj, found)

        # Direita
        if adjacents[3] == 0 and (row, col + 1) not in path and not found:
            found = self.recursive_path_counter(path + [(row, col + 1)], length + 1, obj_len, obj, found)

        return found

    def exists_valid_path_between(self, val1, val2) -> bool:
        """ Verifica se existe caminho válido entre dois valores da Board. """
        obj_len = val2 - val1 - 1
        row, col = self.board.find_number(val1)

        if (row, col) == (None, None):
            row, col = self.board.find_number(val2)
            val2 = val1

        return self.recursive_path_counter([(row, col)], 0, obj_len, val2, False)
    
    def exists_valid_path_between_pos_and_val(self, row, col, val, obj_len) -> bool:
        """ Verifica se existe caminho válido entre uma posicao e um valore da Board. """

        return self.recursive_path_counter([(row, col)], 1, obj_len, val, False)

    def val_in_place(self, val, adjacents) -> bool:
        """ Verifica se o valor está colocado no sítio correto. """
        if val == 1:
            return val + 1 in adjacents

        elif val == self.board.N ** 2:
            return val - 1 in adjacents

        return val + 1 in adjacents and val - 1 in adjacents

    def recursive_unreachable_path(self, explored, row, col) -> bool:
        """ Função auxiliar para verificar espaços que nunca serão preenchidos. """
        adjacents = self.board.get_all_adjacents(row, col)

        for val in adjacents:
            if val is not None and val != 0:
                r, c = self.board.find_number(val)
                val_adjacents = self.board.get_all_adjacents(r, c)

                if not self.val_in_place(val, val_adjacents):
                    return False

        # Abaixo
        if adjacents[0] == 0 and (row + 1, col) not in explored:
            explored += [(row + 1, col)]
            return self.recursive_unreachable_path(explored, row + 1, col)

        # Acima
        if adjacents[1] == 0 and (row - 1, col) not in explored:
            explored += [(row - 1, col)]
            return self.recursive_unreachable_path(explored, row - 1, col)

        # Esquerda
        if adjacents[2] == 0 and (row, col - 1) not in explored:
            explored += [(row, col - 1)]
            return self.recursive_unreachable_path(explored, row, col - 1)

        # Direita
        if adjacents[3] == 0 and (row, col + 1) not in explored:
            explored += [(row, col + 1)]
            return self.recursive_unreachable_path(explored, row, col + 1)

        return True

    def has_unreachable_places(self) -> bool:
        """ Verifica se existem espacos que nunca irão ser preenchidos """
        placed_values = self.board.get_placed_values()
        for row in range(self.board.N):
            for col in range(self.board.N):
                val = self.board.get_number(row, col)

                if val == 0:
                    adjacents = self.board.get_all_adjacents(row, col)
                    if adjacents.count(0) == 1:
                        """ isto piora idk why
                        all_in_place = True
                        for v in adjacents:
                            if v is not None and v != 0:
                                r, c = self.board.find_number(v)
                                if not self.val_in_place(v, self.board.get_all_adjacents(r, c)):
                                    all_in_place = False
                        if all_in_place:
                            for v in adjacents:
                                r, c = self.board.find_number(v)
                                if v is not None and v != 0:
                                    if 1 not in placed_values and self.board.N ** 2 in placed_values:
                                        if not self.exists_valid_path_between_pos_and_val(r, c, min(placed_values), min(placed_values) - 1):
                                            return True
                                    elif self.board.N ** 2 not in placed_values and 1 in placed_values:
                                        if not self.exists_valid_path_between_pos_and_val(r, c, max(placed_values), max(placed_values) - 1):
                                            return True
                                    elif 1 in placed_values and self.board.N ** 2 in placed_values:
                                        return True
                        """
                        if self.recursive_unreachable_path([(row, col)], row, col):
                            return True
                    if adjacents.count(0) == 0:
                        possible_values = self.board.get_possible_values()
                        possible = False
                        for v in adjacents:
                            if v is not None and (v + 1 in possible_values or v - 1 in possible_values):
                                possible = True
                                break
                        if possible == False:
                            return True

        return False

class Board:
    """ Representação interna de um tabuleiro de Numbrix. """
    def __init__(self, N: int, board: dict):
        self.N = N
        self.board = board
        self.placed_values = []

        for place in board:
            bisect.insort(self.placed_values, board[place])

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        if row >= self.N or row < 0 or col >= self.N or col < 0:
            return None

        if (row, col) in self.board:
            return self.board[(row, col)]

        return 0

    def set_number(self, row: int, col: int, value: int) -> None:
        """ Coloca o valor na respetiva posição do tabuleiro. """
        self.board[(row, col)] = value
        bisect.insort(self.placed_values, value)

    def find_number(self, value: int) -> tuple:
        """ Devolve a localização do valor no tabuleiro. """
        for place in self.board:
            if self.board[place] == value:
                return place

        return None, None

    def get_placed_values(self) -> list:
        """ Devolve a lista de valores colocados no tabuleiro. """
        return self.placed_values

    def get_possible_values(self) -> set:
        """ Devolve a lista de valores que não foram ainda colocados no tabuleiro. """
        all_values = set(range(1, self.N ** 2 + 1))
        return all_values.difference(self.board.values())

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple:
        """ Devolve os valores imediatamente abaixo e acima, respectivamente. """
        return self.get_number(row + 1, col), self.get_number(row - 1, col)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple:
        """ Devolve os valores imediatamente à esquerda e à direita, respectivamente. """
        return self.get_number(row, col - 1), self.get_number(row, col + 1)

    def get_all_adjacents(self, row, col) -> tuple:
        """ Devolve os valores imediatamente abaixo, acima, à esquerda e à direita, respetivamente. """
        adjacents = self.adjacent_vertical_numbers(row, col)
        adjacents += self.adjacent_horizontal_numbers(row, col)
        return adjacents

    def get_free_adjacent_positions(self, row, col) -> list:
        """ Devolve todas as posições adjacentes que estejam livres. """
        ret = []

        # Abaixo
        if row + 1 < self.N and self.get_number(row + 1, col) == 0:
            ret += [(row + 1, col)]

        # Acima
        if row > 0 and self.get_number(row - 1, col) == 0:
            ret += [(row - 1, col)]

        # Direita
        if col + 1 < self.N and self.get_number(row, col + 1) == 0:
            ret += [(row, col + 1)]

        # Esquerda
        if col > 0 and self.get_number(row, col - 1) == 0:
            ret += [(row, col - 1)]

        return ret
    
    def recursive_free_area_counter(self, explored, row, col):
        adjacents = self.get_all_adjacents(row, col)

        #abaixo
        if adjacents[0] == 0 and (row + 1, col) not in explored:
            explored += [(row + 1, col)]
            self.recursive_free_area_counter(explored, row + 1, col)
            
        #acima
        if adjacents[1] == 0 and (row - 1, col) not in explored:
            explored += [(row - 1, col)]
            self.recursive_free_area_counter(explored, row - 1, col)
        
        #esquerda
        if adjacents[2] == 0 and (row, col - 1) not in explored:
            explored += [(row, col - 1)]
            self.recursive_free_area_counter(explored, row, col - 1)
        
        #direita
        if adjacents[3] == 0 and (row, col + 1) not in explored:
            explored += [(row, col + 1)]
            self.recursive_free_area_counter(explored, row, col + 1)

        return explored
    
    def get_free_areas(self) -> list:
        ret = []
        explored = []
        for row in range(self.N):
            for col in range(self.N):
                val = self.get_number(row, col)
                to_explore = []
                if val == 0 and (row, col) not in explored:
                    to_explore = [(row, col)]
                    ret += [self.recursive_free_area_counter(to_explore, row, col)]
                    explored += to_explore

        return ret

    def get_frontier(self, free_area) -> list:
        ret = []
        for val in free_area:
            for v in self.get_all_adjacents(val[0], val[1]):
                if v is not None and v != 0:
                    ret += [v]
        return ret

    def get_copy(self):
        """ Devolve uma cópia da Board. """
        copy_board = dict()
        for place in self.board:
            copy_board[place] = self.board[place]

        return Board(self.N, copy_board)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna uma instância da classe Board. """
        with open(filename, 'r') as file:
            N = int(file.readline()[:-1])
            board_tmp = list(map(lambda x: list(map(int, x[:-1].split('\t'))), file.readlines()))
            board = dict()

            for row in range(N):
                for col in range(N):
                    if board_tmp[row][col] != 0:
                        board[(row, col)] = board_tmp[row][col]

        return Board(N, board)

    def __repr__(self) -> str:
        ret = ""

        for row in range(self.N):
            for col in range(self.N):
                if (row, col) in self.board:
                    ret += str(self.board[(row, col)])

                else:
                    ret += '0'

                ret += '\t'

            ret = ret[:-1] + '\n'

        return ret

    def to_string(self) -> str:
        """ Retorna representação em string da Board. É igual a __repr__, para compatibilidade com os exemplos do
        enunciado. """
        return self.__repr__()


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        super().__init__(NumbrixState(board))

    def actions(self, state: NumbrixState) -> list:
        """ Retorna uma lista de ações que podem ser executadas a partir do estado passado como argumento. """
        ret = []

        board = dict(sorted(state.board.board.items(), key=lambda item: item[1]))
        possible_values = state.board.get_possible_values()

        for place in board:
            row, col = place
            value = state.board.get_number(row, col)
            free_positions = state.board.get_free_adjacent_positions(row, col)
            adjacents = state.board.get_all_adjacents(row, col)

            if len(free_positions) == 1 and not state.val_in_place(value, adjacents):
                if value + 1 in possible_values:
                    return [(free_positions[0][0], free_positions[0][1], value + 1)]

                if value - 1 in possible_values:
                    return [(free_positions[0][0], free_positions[0][1], value - 1)]

                else:
                    return []

            for position in free_positions:
                r, c = position
                adjacents = state.board.get_all_adjacents(r, c)

                for v in adjacents:
                    if v is not None and v != 0 and v != value:
                        if abs(v - value) == 2:
                            if max(v, value) - 1 in possible_values:
                                if adjacents.count(0) == 0:
                                    return [(r, c, max(v, value) - 1)]

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

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """ Retorna o estado resultante de executar a 'action' sobre 'state' passado como argumento. A ação a
        executar deve ser uma das presentes na lista obtida pela execução de self.actions(state). """
        copy_board = state.board.get_copy()
        row, col, value = action
        copy_board.set_number(row, col, value)

        return NumbrixState(copy_board)

    def goal_test(self, state: NumbrixState) -> bool:
        """ Retorna True se e só se o estado passado como argumento é um estado objetivo. Deve verificar se todas as
        posições do tabuleiro estão preenchidas com uma sequência de números adjacentes. """
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

            # Abaixo
            if adjacents[0] == i:
                row += 1

            # Acima
            elif adjacents[1] == i:
                row -= 1

            # Esquerda
            elif adjacents[2] == i:
                col -= 1

            # Direita
            elif adjacents[3] == i:
                col += 1

            else:
                return False

        return True

    def h(self, node: Node) -> int or float:
        """ Função heurística. """
        action = node.action
        state = node.state

        if action is not None:
            adjacents = state.board.get_all_adjacents(action[0], action[1])

            # best option
            if adjacents.count(0) == 0:
                if action[2] != 1 and action[2] + 1 in adjacents and action[2] - 1 in adjacents:
                    return -math.inf

                if action[2] == 1 and action[2] + 1 in adjacents:
                    return -math.inf

                if action[2] == state.board.N ** 2 and action[2] - 1 in adjacents:
                    return -math.inf

            if action[2] != 1 and action[2] + 1 in adjacents and action[2] - 1 in adjacents:
                r_s, c_s = state.board.find_number(action[2] + 1)
                free_s = state.board.get_free_adjacent_positions(r_s, c_s)
                r_a, c_a = state.board.find_number(action[2] - 1)
                free_a = state.board.get_free_adjacent_positions(r_a, c_a)
                free = set(free_s).intersection(free_a)

                if len(free) > 0:
                    return self.initial.board.N ** 2 - len(state.board.get_placed_values())

                return -math.inf

            possible_values = state.board.get_possible_values()

            if adjacents.count(0) == 0 and (action[2] + 1 in possible_values or action[2] - 1 in possible_values):
                return math.inf

            for val in adjacents:
                if val is not None and val != 0:
                    row, col = state.board.find_number(val)
                    val_adjacents = state.board.get_all_adjacents(row, col)

                    if val_adjacents.count(0) == 0 and (val + 1 in possible_values or val - 1 in possible_values):
                        return math.inf

                    if val_adjacents.count(0) == 1 and val + 1 in possible_values and val - 1 in possible_values:
                        return math.inf

            if state.has_unreachable_places():
                return math.inf

            if state.has_impossible_free_areas():
                return math.inf

            placed_values = state.board.get_placed_values()

            for i in range(len(placed_values) - 1):
                if not state.exists_valid_path_between(placed_values[i], placed_values[i + 1]):
                    return math.inf

            if 1 not in placed_values and not state.exists_valid_path_between(1, placed_values[0]):
                return math.inf

            if state.board.N ** 2 not in placed_values:
                if not state.exists_valid_path_between(placed_values[-1], state.board.N ** 2):
                    return math.inf

        return self.initial.board.N ** 2 - len(state.board.get_placed_values())


if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    board = Board.parse_instance(sys.argv[1])

    problem = Numbrix(board)
    goal_node = greedy_search(problem)
    print(goal_node.state.board.to_string(), end="")
