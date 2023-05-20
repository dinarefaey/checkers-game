from collections import Counter
import random
from typing import Callable, List, Tuple
from copy import deepcopy

Board = List[List[int]]
Position = Tuple[int, int]
Positions = List[Position]
Moves = List[Tuple[Position, Positions]]


class Checkers(object):
    """
    checkers class contains methods to play checkers
    """

    WHITE = 1
    WHITE_MAN = 1
    WHITE_KING = 3
    BLACK = 0
    BLACK_MAN = 2
    BLACK_KING = 4
    DX = [1, 1, -1, -1]
    DY = [1, -1, 1, -1]
    OO = 10 ** 9
# __init__
    def __init__(self, size: int = 8) -> None:
        if size % 2 != 0 or size < 4:
            raise Exception("The size of the board must be even and greater than 3")

        self.size = size
        self.board = []
        piece = self.WHITE_MAN
        for i in range(size):
            l = []
            f = i % 2 == 1
            if i == size / 2 - 1:
                piece = 0
            elif i == size / 2 + 1:
                piece = self.BLACK_MAN
            for _ in range(size):
                if f:
                    l.append(piece)
                else:
                    l.append(0)
                f = not f
            self.board.append(l)

        self.stateCounter = Counter()

    def EncodeBoardofgame(self) -> int:  
        """
            int: the count of the encoded game board
        """
        count = 0
        size1 = self.size
        for i in range(size1):
            for j in range(size1):
                # make the minimum count = 5,
                # so that it's greater than greatest count of the board (4)
                num = i * size1
                num = num + (j + 5)
                count = count + (num * self.board[i][j])
        return count

    def getBoard(self):
        """
            Board: game board
        """
        return deepcopy(self.board)

    def setBoard(self, board: Board):
        """
            board (Board): board to set the game borad to
        """
        self.board = deepcopy(board)

    def isValid(self, n: int, m: int) -> bool:
        """
            bool: the given position is valid
        """
        bool1 = (n >= 0)
        bool2 = (n < self.size)
        bool3 = (m >= 0)
        bool4 = (m < self.size)
        return bool1 and bool2 and bool3 and bool4

    def nextPositions(self, p1: int, p2: int):
        Tuple[Positions, Positions]

        if self.board[p1][p2] == 0:
            return []

        player = self.board[p1][p2] % 2
        capt_moves = []
        norm_moves = []
        s = 1 if player == self.WHITE else -1
        ran = 2 if self.board[p1][p2] <= 2 else 4
        for i in range(ran):
            x = p1 + s * self.DX[i]
            y = p2 + s * self.DY[i]
            if self.isValid(x, y):
                if self.board[x][y] == 0:
                    norm_moves.append((x, y))
                elif self.board[x][y] % 2 == 1 - player:
                    x += s * self.DX[i]
                    y += s * self.DY[i]
                    if self.isValid(x, y) and self.board[x][y] == 0:
                        capt_moves.append((x, y))

        return norm_moves, capt_moves

    def nextMoves(self, player: int) -> Moves:
        """
            Moves: valid moves for the player
        """
        captureMoves = []
        normalMoves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != 0 and self.board[x][y] % 2 == player:
                    normal, capture = self.nextPositions(x, y)
                    if len(normal) != 0:
                        normalMoves.append(((x, y), normal))
                    if len(capture) != 0:
                        captureMoves.append(((x, y), capture))
        if len(captureMoves) != 0:
            return captureMoves
        return normalMoves

    def playMove(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[bool, int, bool]:
        """
            canCapture (bool): if the player can capture more pieces.
           cancel (int): the removed piece (if any).
            promoted (bool) if the current piece is promoted).
        """
        self.board[x2][y2] = self.board[x1][y1]
        self.board[x1][y1] = 0

        cancel = 0
        if abs(x2 - x1) == 2:  # capture move
            x3 = x2 - x1
            y3 = y2 - y1
            cancel = self.board[x1 + x3 // 2][y1 + y3 // 2]
            self.board[x1 + x3 // 2][y1 + y3 // 2] = 0  # remove captured piece

        # promote to king
        if self.board[x2][y2] == self.WHITE_MAN and x2 == self.size - 1:
            self.board[x2][y2] = self.WHITE_KING
            return False, cancel, True
        if self.board[x2][y2] == self.BLACK_MAN and x2 == 0:
            self.board[x2][y2] = self.BLACK_KING
            return False, cancel, True

        if abs(x2 - x1) != 2:
            return False, cancel, False

        return True, cancel, False

    def undoMove(self, x1: int, y1: int, x2: int, y2: int, cancel=0, tmp=False):
        """Undo a move and return the board to its previous state
        """
        if tmp:
            bool1 = self.board[x2][y2] == self.WHITE_KING
            bool2 = self.board[x2][y2] == self.BLACK_KING
            if bool1:
                self.board[x2][y2] = self.WHITE_MAN

            if bool2:
                self.board[x2][y2] = self.BLACK_MAN

        self.board[x1][y1] = self.board[x2][y2]
        self.board[x2][y2] = 0

        if abs(x2 - x1) == 2:
            x3 = x2 - x1
            y3 = y2 - y1
            self.board[x1 + x3 // 2][y1 + y3 // 2] = cancel

    def evaluate1(self, max: int) -> int:
        rows: int = 8
        cols: int = 8

        score = 0
        for i in range(rows):
            for j in range(cols):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == max:
                        score += (self.board[i][j] + 1) // 2
                    else:
                        score -= (self.board[i][j] + 1) // 2
        return score * 1000

    def ContainsPlayer(self, x: int, y: int, player: int) -> bool:#10
        """return if cell at (x, y) contains player
            bool: if cell at (x, y) contains player
        """
        return self.board[x][y] != 0 and self.board[x][y] % 2 == player

    def endGame(self, maximizer: int) -> int:
        """evaluate the current state of the board based on end game strategies
            between maximizer player and the opponent
        """
        num1 = 0
        num2 = 0
        maxnumpic = 0
        minnumpic = 0
        row = 0
        tmp = 0 if maximizer == self.WHITE else self.size - 1
        minimizer = 1 - maximizer
        minPositions = []
        value = self.size
        for i in range(value):
            for j in range(value):
                if self.cellContains(i, j, minimizer):
                    minPositions.append((i, j))

        value2 = (self.board[i][j] + 1) // 2
        for i in range(value):
            for j in range(value):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == maximizer:
                        maxnumpic = maxnumpic + 1
                        if value2 == 1:
                            row = row + abs(tmp - i)
                        num1 = num1 + value2
                        for x, y in minPositions:
                            m1 = x - i
                            m2 = y - j
                            num2 = num2 + m1 * 2 + m2 * 2
                    else:
                        minnumpic = minnumpic + 1
                        num1 = num1 - value2

        # penalize if the minimizer is in the corner to be able to trap him at the end of the game
        minCorner = 0
        for x, y in minPositions:
            if (x == 0 and y == 1) or (x == 1 and y == 0) or (x == self.size - 1 and y == self.size - 2) \
                    or (x == self.size - 2 and y == self.size - 1):
                minCorner = 1

        maxCorner = 0
        if self.cellContains(0, 1, maximizer) or self.cellContains(1, 0, maximizer) \
                or self.cellContains(self.size - 1, self.size - 2, maximizer) \
                or self.cellContains(self.size - 2, self.size - 1, maximizer):
            maxCorner = 1
        if maxnumpic > minnumpic:
            bool1 = True
        else:
            bool1 = False
        if bool1:  # come closer to opponent
            nnn = num1 * 1000 - num2 - minCorner * 5 + row * 10
            return
        else:  # run away
            return num1 * 1000 + num2 + maxCorner * 5

    def evaluate2(self, maximizer: int) -> int:
        """evaluate the current state of the board
        """

        sold = 0
        k = 0
        Row = 0
        midBox = 0
        midRow = 0
        value = 0
        prot = 0
        val = self.size
        for i in range(val):
            for j in range(val):
                if self.board[i][j] != 0:
                    sign = 1 if self.board[i][j] % 2 == maximizer else -1
                    if self.board[i][j] <= 2:
                        sold = sold + sign * 1
                    else:
                        k = k + sign * 1
                    bool1 = (i == 0 and maximizer == self.WHITE)
                    bool2 = (i == self.size - 1 and maximizer == self.BLACK)
                    if sign == 1 and (bool1 or bool2):
                        Row = Row + 1
                    bool3 = i == self.size / 2 - 1
                    bool4 = i == self.size / 2
                    if bool3 or bool4:
                        if j >= self.size / 2 - 2 and j < self.size / 2 + 2:
                            midBox = midBox + sign * 1
                        else:
                            midBox = midBox + sign * 1

                    Dir = 1 if maximizer == self.WHITE else -1
                    bool5 = False
                    for v in range(4):
                        x = i + self.DX[v]
                        y = j + self.DY[v]
                        nn = i - self.DX[v]
                        mm = j - self.DY[v]
                        Dir2 = abs(x - nn) / (x - nn)
                        # b1=self.board[x][y]
                        # b2=self.board[nn][mm]

                        if self.isValid(x, y) and self.board[x][y] != 0 and self.board[x][y] % 2 != maximizer \
                                and self.isValid(nn, mm) and self.board[nn][mm] == 0 and (
                                self.board[x][y] > 2 or Dir != Dir2):
                            bool5 = True
                            break

                    if bool5:
                        value = value + sign * 1
                    else:
                        prot = prot + sign * 1
        result = sold * 2000 + k * 4000 + Row * 400 + midBox * 250 + midRow * 50 - 300 * value + 300 * prot
        return result

    def stateValue(self, max: int) -> int:
        maxp = 0
        minp = 0
        rows = 8
        cols = 8
        for i in range(rows):
            for j in range(cols):
                if self.board[i][j] != 0:
                    if self.board[i][j] % 2 == max:
                        maxp += 1
                    else:
                        minp += 1
        if (maxp > minp):
            return -self.stateCounter[self.EncodeBoardofgame()]
        return 0

    def minimax(
            self,
            person: int,
            Max: int,
            Len: int = 0,
            AlphA: int = -OO,
            BetA: int = OO,
            maxDepth: int = 4,
            evaluate: Callable[[int], int] = evaluate2,
            moves: Moves = None,
    ) -> int:
        """Get the score of the board using alpha-beta algorithm

        """
        if moves == None:
            moves = self.nextMoves(person)
        if len(moves) == 0 or Len == maxDepth:
            target = evaluate(self, Max)
            # if there is no escape from losing, maximize number of moves to lose
            if target < 0:
                target = target + Len
            return target

        Best = -self.OO
        bool1 = person != Max
        if bool1:
            Best = self.OO

        # sort moves by the minimum next positions
        moves.sort(key=lambda move: len(move[1]))
        num1 = moves
        for position in num1:
            x1, y1 = position[0]
            for x2, y2 in position[1]:

                Capture, cancel, tmp = self.playMove(x1, y1, x2, y2)
                bool2 = False

                if Capture:
                    _, Capture2 = self.nextPositions(x2, y2)
                    if len(Capture2) != 0:
                        bool2 = True
                        numofMoves = [((x2, y2), Capture2)]

                        if person == Max:
                            v1 = self.minimax(person, Max, Len + 1, AlphA, BetA, maxDepth, evaluate, numofMoves)
                            Best = max(
                                Best,
                                v1
                            )
                            AlphA = max(AlphA, Best)
                        else:
                            v1 = self.minimax(person, Max, Len + 1, AlphA, BetA, maxDepth, evaluate, numofMoves)
                            Best = min(
                                Best,
                                v1
                            )
                            BetA = min(BetA, Best)

                if not bool2:

                    if person == Max:
                        v2 = self.minimax(1 - person, Max, Len + 1, AlphA, BetA, maxDepth, evaluate)
                        Best = max(
                            Best,
                            v2
                        )
                        AlphA = max(AlphA, Best)
                    else:
                        v2 = self.minimax(1 - person, Max, Len + 1, AlphA, BetA, maxDepth, evaluate)
                        Best = min(
                            Best,
                            v2
                        )
                        BetA = min(BetA, Best)

                self.undoMove(x1, y1, x2, y2, cancel, tmp)

                if BetA <= AlphA:
                    break
            if BetA <= AlphA:
                break

        return Best
    def minimaxPlay(  # 14
            self,
            person: int,
            moves: Moves = None,
            maxDepth: int = 4,
            evaluate: Callable[[int], int] = evaluate2,
            Printit: bool = True,
    ) -> Tuple[bool, bool]:
        """play a move using minimax algorithm
            if the player should continue capturing, it will
        """

        if moves == None:
            moves = self.nextMoves(person)
        if len(moves) == 0:
            if Printit:
                print(("WHITE" if person == self.BLACK else "BLACK") + " Player wins")
            return False, False

        self.stateCounter[self.EncodeBoardofgame()] = self.stateCounter[self.EncodeBoardofgame()] + 1

        random.shuffle(moves)
        bestValue = -self.OO
        # bestMove = None
        Bestmv = None

        for position in moves:
            x1, y1 = position[0]
            for x2, y2 in position[1]:
                _, cancel, tmp = self.playMove(x1, y1, x2, y2)
                v1 = self.minimax(1 - person, person, maxDepth=maxDepth, evaluate=evaluate)
                v1 = v1 + 2 * self.stateValue(person)
                self.undoMove(x1, y1, x2, y2, cancel, tmp)
                if v1 > bestValue:
                    bestValue = v1
                    Bestmv = (x1, y1, x2, y2)

        x1, y1, x2, y2 = Bestmv
        if Printit:
            print(f"Move from ({x1}, {y1}) to ({x2}, {y2})")
        Capture, cancel, _ = self.playMove(x1, y1, x2, y2)
        if Printit:
            self.printBoard(x2, y2)

        if Capture:
            _, CAPTURE = self.nextPositions(x2, y2)
            if len(CAPTURE) != 0:
                self.minimaxPlay(person, [((x2, y2), CAPTURE)], maxDepth, evaluate, Printit)

        self.stateCounter[self.EncodeBoardofgame()] = self.stateCounter[self.EncodeBoardofgame()] + 1
        RESULT = cancel != 0
        return True, RESULT

    """ 
    def minimaxx(self, board, player, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')): #minmax algorithm
        if depth == 0 or self.isGameOver(board):
            return self.evaluate(board, player)

        if maximizing_player:
            bestValue = float('-inf')
            moves = self.getValidMoves(board, player)
            for move in moves:
                new_board = self.makeMove(board, move)
                value = self.minimax(new_board, player, depth - 1, False, alpha, beta)
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                if beta <= alpha:
                    break
            return bestValue
        else:
            bestValue = float('inf')
            moves = self.getValidMoves(board, player)
            for move in moves:
                new_board = self.makeMove(board, move)
                value = self.minimax(new_board, player, depth - 1, True, alpha, beta)
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break
            return bestValue

    def play(self, board, player, depth=4):
        best_move = None
        best_score = float('-inf')

        valid_moves = self.getValidMoves(board, player)
        for move in valid_moves:
            new_board = self.makeMove(board, move)
            score = self.minimax(new_board, depth, False)
            if score > best_score:
                best_score = score
                best_move = move

        # Perform the best move on the board
        if best_move is not None:
            updated_board = self.makeMove(board, best_move)
            return updated_board

        # No valid moves found
        return board

    def minimax(self, board, depth, maximizing_player):
        if depth == 0 or self.isGameOver(board):
            return self.evaluate(board)

        if maximizing_player:
            best_value = float('-inf')
            valid_moves = self.getValidMoves(board, self.WHITE)
            for move in valid_moves:
                new_board = self.makeMove(board, move)
                value = self.minimax(new_board, depth - 1, False)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            valid_moves = self.getValidMoves(board, self.BLACK)
            for move in valid_moves:
                new_board = self.makeMove(board, move)
                value = self.minimax(new_board, depth - 1, True)
                best_value = min(best_value, value)
            return best_value
    
    
    
    
    
    
    
    
    
    
    """