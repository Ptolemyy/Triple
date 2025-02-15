from manim import *
import json
import numpy as np

class FilledGrids(Scene):
    def whole_board(self, board, pool):
        non_zero_values = np.reshape(board, 16)

        grid_4 = VGroup(*[Square(side_length=1) for _ in range(16)]).arrange_in_grid(
            rows=4, cols=4, buff=0
        )
        grid_4.shift(-grid_4.get_corner(DL))

        index = 0
        for i, square in enumerate(grid_4):
            row, col = divmod(i, 4)
            if board[row][col] != 0:
                green_value = int(non_zero_values[index])
                text = Text(str(green_value), font_size=24)
                text.move_to(square.get_center())
                grid_4.add(text)
            index += 1

            square.set_fill(GREEN, opacity=1)
            square.set_stroke(WHITE, width=1)

        grid_12 = VGroup(*[Square(side_length=1) for _ in range(2)]).arrange(RIGHT, buff=0)
        for i, square in enumerate(grid_12):
            square.set_fill(BLUE, opacity=1)
            square.set_stroke(WHITE, width=1)
            blue_value = int(pool[i])
            blue_text = Text(str(blue_value), font_size=24)
            blue_text.move_to(grid_12[i].get_center())
            grid_12.add(blue_text)

        displacement = grid_4.get_corner(UL) - grid_12.get_corner(DR)
        grid_12.shift(displacement + RIGHT * 2)

        all_grids = VGroup(grid_4, grid_12)
        all_grids.move_to(ORIGIN)
        return all_grids
    def construct(self):
        data_path = "data/"
        animation_name = "Wed_Feb__5_08_28_27_2025of_model_172.json"

        with open(data_path + animation_name) as f:
            datas = json.loads(f.read())
        board_and_pool = [data["board"][0] for data in datas]

        for frames in board_and_pool:
            frames = np.array(frames)
            pool = np.copy([frames[0][0], frames[0][1]])
            board = np.copy(frames[1:-1,1:-1])
            single_frame = self.whole_board(board, pool)
            self.add(single_frame)
            self.wait(0.3)
            self.remove(single_frame)



