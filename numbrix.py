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
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

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

    def find_mininum(self) -> tuple[int | None, int | None, int]:
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
        self.board = board

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        max_row, max_col, maximum = state.board.find_maximum()
        min_row, min_col, minimum = state.board.find_mininum()

        # TODO
        pass

    def result(self, state: NumbrixState, action) -> NumbrixState:
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """

        # FIXME adicionar verificação de action, ainda não sei bem o que pôr
        copy_board = [line[:] for line in state.board.board]
        row, col, value = action
        copy_board[row][col] = value
        return NumbrixState(copy_board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. """

        i = 1
        row, col = state.board.find_number(i)

        if row == None or col == None:
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
        # TODO
        pass

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

    # board = Board.parse_instance(sys.argv[1])

    # i1.txt do enunciado
    # board = Board(3, [[0,0,0],[0,0,2],[0,6,0]])

    board = Board(3, [[9,4,3],[8,5,2],[7,6,1]])
    # Criar uma instância de Numbrix:
    problem = Numbrix(board)
    # Criar um estado com a configuração inicial:
    s = NumbrixState(board)
    print("Is goal?", problem.goal_test(s))
    print("Solution:\n", s.board.to_string(), sep="")
