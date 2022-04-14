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
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search
from utils import unique
from itertools import chain, combinations #TODO podemos importar isto? (é usado no utils)

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        s_size = self.longest_chain_size()
        o_size = other.longest_chain_size()
        if s_size != o_size:
            return s_size > o_size
        return self.id < other.id

    def recursive_chain_counter(self, explored, row, col, val):
        ret = 1
        adjacents = self.board.adjacent_vertical_numbers(row, col) 
        adjacents += self.board.adjacent_horizontal_numbers(row, col)
        if val+1 in adjacents and val+1 not in explored:
            new_row, new_col = self.board.find_number(val+1)
            explored += [val+1]
            ret = 1 + self.recursive_chain_counter(explored, new_row, new_col, val+1)
        if val-1 in adjacents and val-1 not in explored:
            new_row, new_col = self.board.find_number(val-1)
            explored += [val-1]
            ret = 1 + self.recursive_chain_counter(explored, new_row, new_col, val-1)
        return ret

    def longest_chain_size(self) -> int:
        max = 0
        explored = []
        for row in range(self.board.N):
            for col in range(self.board.N):
                val = self.board.get_number(row, col)
                if (val != 0):
                    explored += [val]
                    ret = self.recursive_chain_counter(explored, row, col, val)
                    if ret > max:
                        max = ret
                if max > self.board.N**2:
                    return max
        return max


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

    def find_number(self, value: int) -> tuple[int, int] | tuple[None, None]:
        """ Deveolve a localização do valor no tabuleiro. """
        for rows in range(self.N):
            for cols in range(self.N):
                if self.board[rows][cols] == value:
                    return (rows, cols)

        return None, None

    def possible_values(self) -> set:
        """ Devolve a lista de valores que não foram ainda colocados
        no tabuleiro """
        all_values = set(range(1, self.N**2 + 1))
        placed_values = set(v for row in self.board for v in row)
        return all_values.difference(placed_values)

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

        return (row, col, minimum)

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

        return (row, col, maximum)

    def adjacent_vertical_numbers(self, row: int, col: int) -> tuple[int | None, int | None]:
        """ Devolve os valores imediatamente abaixo e acima,
        respectivamente. """
        return self.get_number(row + 1, col), self.get_number(row - 1, col)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> tuple[int | None, int | None]:
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        return self.get_number(row, col - 1), self.get_number(row, col + 1)

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

        #TODO é possível por isto a cortar jogadas q nao vao fazer sentido no futuro
        # mas dá um bcd trabalho e se calhar a heuristica faz isto / é suficiente

        if (action[0] < 0 or action[0] >= state.board.N
                or action[1] < 0 or action[1] >= state.board.N):
            return False

        adjacents = state.board.adjacent_vertical_numbers(action[0], action[1]) 
        adjacents += state.board.adjacent_horizontal_numbers(action[0], action[1])

        if ((action[2] != state.board.N and action[2]+1 not in adjacents and state.board.find_number(action[2]+1) != (None, None))
                or (action[2] != 1 and action[2]-1 not in adjacents and state.board.find_number(action[2]-1) != (None, None))):
            return False

        if (action[2] in adjacents
                or action[2] > state.board.N**2
                or action[2] < 1
                or state.board.find_number(action[2]) != (None, None)):
            return False
        
        if(action[2] != 1 
                and action[2] != state.board.N
                and adjacents.count(0) == 0 
                and (action[2]-1 not in adjacents or action[2]+1 not in adjacents)):
            return False
        
        new_state = self.result(state, action)
        possible_values = new_state.board.possible_values()

        for val in adjacents:
            if (val is not None):
                row, col = new_state.board.find_number(val)
                val_adjacents = new_state.board.adjacent_vertical_numbers(row, col) 
                val_adjacents += new_state.board.adjacent_horizontal_numbers(row, col)
                if (val_adjacents.count(0) == 0 and ((val+1 in possible_values and val+1 not in val_adjacents) 
                        or (val-1 in possible_values and val-1 not in val_adjacents))):
                    return False

        for row in range(new_state.board.N):
            for col in range(new_state.board.N):
                val = new_state.board.get_number(row, col)
                if val == 0:
                    adjacents = new_state.board.adjacent_vertical_numbers(row, col) 
                    adjacents += new_state.board.adjacent_horizontal_numbers(row, col)
                    if (adjacents.count(0) == 0):
                        pairs = list(chain.from_iterable(combinations(adjacents, r) for r in range(2,3)))[1:]
                        for pair in pairs:
                            if (pair[1] is None or pair[0] is None):
                                continue
                            if (pair[1] - pair[0] == 2 and pair[1]-1 not in possible_values):
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
        for row in range(state.board.N):
            for col in range(state.board.N):
                value = state.board.get_number(row, col)
                if (value == 0):
                    adjacents = state.board.adjacent_vertical_numbers(row, col)
                    adjacents += state.board.adjacent_horizontal_numbers(row, col)
                    
                    if (adjacents.count(0) == 0):
                        pairs = list(chain.from_iterable(combinations(adjacents, r) for r in range(2,3)))[1:]
                        add_to_ret = []
                        for pair in pairs:
                            if (pair[1] is None or pair[0] is None):
                                continue
                            if (pair[1] - pair[0] == 2 and self.is_valid_action(state, (row, col, pair[1]-1))):
                                return [(row, col, pair[1]-1)]
                            
                        else:
                            test = [i + 1 for i in adjacents if i != None and i != 0 and i != state.board.N]
                            test += [i - 1 for i in adjacents if i != None and i != 0 and i != 1]                            
                            for val in test:
                                if (val in possible_vals and self.is_valid_action(state, (row, col, val))):
                                    add_to_ret += [(row, col, val)]
                            if adjacents.count(0) == 0:
                                return unique(add_to_ret)

                    test = [i + 1 for i in adjacents if i != None and i != 0 and i != state.board.N]
                    test += [i - 1 for i in adjacents if i != None and i != 0 and i != 1]
                    for val in test:
                        if (val in possible_vals and self.is_valid_action(state, (row, col, val))):
                            ret += [(row, col, val)]
        return unique(ret)

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """

        copy_board = [line[:] for line in state.board.board]
        row, col, value = action
        copy_board[row][col] = value
        return NumbrixState(Board(state.board.N, copy_board))

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
        if(action is not None):
            action_adjacents = state.board.adjacent_horizontal_numbers(action[0], action[1])
            action_adjacents += state.board.adjacent_vertical_numbers(action[0], action[1])
            #best option
            if(action[2]+1 in action_adjacents and action[2]-1 in action_adjacents):
                return 0
            if((action[2]+1 in action_adjacents or action[2]-1 in action_adjacents) and action_adjacents.count(0) == 0):
                return 0
        #TODO reduce complexity
        return self.initial.board.N**2 - state.longest_chain_size()

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
