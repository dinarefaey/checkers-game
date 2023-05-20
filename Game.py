import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from Checkers import Checkers, Positions
from enum import Enum
import time
window = tk.Tk()
window.title("Checkers")
IMG_SIZE = 60
black_man_img = ImageTk.PhotoImage(Image.open('assets/black_man.png').resize((IMG_SIZE, IMG_SIZE)))
black_king_img = ImageTk.PhotoImage(Image.open('assets/black_king.png').resize((IMG_SIZE, IMG_SIZE)))
white_man_img = ImageTk.PhotoImage(Image.open('assets/white_man.png').resize((IMG_SIZE, IMG_SIZE)))
white_king_img = ImageTk.PhotoImage(Image.open('assets/white_king.png').resize((IMG_SIZE, IMG_SIZE)))
blank_img = ImageTk.PhotoImage(Image.open('assets/blank.png').resize((IMG_SIZE, IMG_SIZE)))


class Mode(Enum):
    SINGLE_PLAYER = 0
    MULTIPLE_PLAYER = 1


class Algorithm(Enum):
    MINIMAX = 1
    RANDOM = 0


CHECKER_SIZE = 8
GAME_MODE = Mode.MULTIPLE_PLAYER  # Set to MULTIPLE_PLAYER
STARTING_PLAYER = Checkers.BLACK
USED_ALGORITHM = Algorithm.MINIMAX
MAX_DEPTH = 5
EVALUATION_FUNCTION = Checkers.evaluate2
INCREASE_DEPTH = True


def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    r, g, b = rgb
    return f'#{r:02x}{g:02x}{b:02x}'


class GUI:

    def __init__(self) -> None:
        super().__init__()
        self.game = Checkers(CHECKER_SIZE)
        self.history = [self.game.getBoard()]
        self.historyPtr = 0

        self.maxDepth = MAX_DEPTH

        self.player = STARTING_PLAYER
        self.btn = [[None] * self.game.size for _ in range(self.game.size)]  # Initialize btn attribute

        frm_board = tk.Frame(master=window)
        frm_board.pack(fill=tk.BOTH, expand=True)
        for i in range(self.game.size):
            frm_board.columnconfigure(i, weight=1, minsize=IMG_SIZE)
            frm_board.rowconfigure(i, weight=1, minsize=IMG_SIZE)

            for j in range(self.game.size):
                frame = tk.Frame(master=frm_board)
                frame.grid(row=i, column=j, sticky="nsew")

                self.btn[i][j] = tk.Button(master=frame, width=IMG_SIZE, height=IMG_SIZE, relief=tk.FLAT)
                self.btn[i][j].pack(expand=True, fill=tk.BOTH)

        self.update()
        nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
        self.highlight(nextPositions)
        self.computer_vs_computer()  # Start the game with computer vs computer
        window.mainloop()






    def update(self):
        # Update the GUI board based on the game state
        for i in range(self.game.size):
            f = i % 2 == 1
            for j in range(self.game.size):
                if f:
                    self.btn[i][j]['bg'] = 'gray30'
                else:
                    self.btn[i][j]['bg'] = 'white'
                img = blank_img
                if self.game.board[i][j] == Checkers.BLACK_MAN:
                    img = black_man_img
                elif self.game.board[i][j] == Checkers.BLACK_KING:
                    img = black_king_img
                elif self.game.board[i][j] == Checkers.WHITE_MAN:
                    img = white_man_img
                elif self.game.board[i][j] == Checkers.WHITE_KING:
                    img = white_king_img

                self.btn[i][j]["image"] = img
                f = not f
        window.update()

    def highlight(self, positions: Positions):
        # Highlight the valid positions on the GUI board
        for x in range(self.game.size):
            for y in range(self.game.size):
                defaultbg = self.btn[x][y].cget('bg')
                self.btn[x][y].master.config(highlightbackground=defaultbg, highlightthickness=3)

        for position in positions:
            x, y = position
            self.btn[x][y].master.config(highlightbackground="yellow", highlightthickness=3)





    def computer_vs_computer(self):
        if GAME_MODE == Mode.MULTIPLE_PLAYER:
            moves_without_capture = 0
            max_moves_without_capture = 100

            while moves_without_capture < max_moves_without_capture:
                if self.player == Checkers.WHITE:
                    if USED_ALGORITHM == Algorithm.MINIMAX:
                        self.game.minimaxPlay(1 - self.player, maxDepth=self.maxDepth, evaluate=EVALUATION_FUNCTION,
                                              Printit=False)
                    elif USED_ALGORITHM == Algorithm.RANDOM:
                        self.game.randomPlay(1 - self.player, enablePrint=False)
                    self.update()
                    self.player = 1 - self.player
                    nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
                    self.highlight(nextPositions)
                    if len(nextPositions) == 0:
                        winner = "BLACK" if self.player == Checkers.WHITE else "WHITE"
                        messagebox.showinfo(message=f"{winner} Player won!", title="Checkers")
                        break
                    moves_without_capture += 1
                    time.sleep(1)  # Delay of 4 seconds
                else:
                    if USED_ALGORITHM == Algorithm.MINIMAX:
                        self.game.minimaxPlay(1 - self.player, maxDepth=self.maxDepth, evaluate=EVALUATION_FUNCTION,
                                              Printit=False)
                        # elif USED_ALGORITHM == Algorithm.RANDOM:
                    #    self.game.randomPlay(1 - self.player, enablePrint=False)
                    self.update()
                    self.player = 1 - self.player
                    nextPositions = [move[0] for move in self.game.nextMoves(self.player)]
                    self.highlight(nextPositions)
                    if len(nextPositions) == 0:
                        winner = "BLACK" if self.player == Checkers.WHITE else "WHITE"
                        messagebox.showinfo(message=f"{winner} Player won!", title="Checkers")
                        break
                    moves_without_capture += 1
                  #  time.sleep(1)  # Delay of 4 seconds
            else:
                messagebox.showinfo(message="Draw!", title="Checkers")


GUI()
