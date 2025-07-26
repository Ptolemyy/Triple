import random
from flask import Flask, render_template, request, jsonify, session
import numpy as np
from Triple import Triple
import json # 用于保存文件
from datetime import datetime # 用于生成唯一文件名
import os # 用于创建目录
 
# 1. 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'a-very-secret-and-random-key-for-session'
 
# 定义一个保存游戏记录的目录
SAVE_DIR = "game_records"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
 
@app.route('/')
def index():
    if 'main_grid' not in session:
        print("New session started. Initializing game data.")
        # 初始化游戏数据，并新增 click_history
        session['top_grid'] = [1, 1]
        session['main_grid'] = np.zeros(16, dtype=int).tolist()
        session['click_history'] = [] # 新增：初始化点击历史记录
        session['grid_history'] = [] # 新增：初始化格子历史记录
        
    return render_template('index.html', top_grid=session['top_grid'], main_grid=session['main_grid'])
 
@app.route('/click', methods=['POST'])
def handle_click():
    data = request.get_json()
    clicked_index = data.get('index')
    if 'main_grid' not in session:
        return jsonify({'error': 'Session not found. Please refresh the page.'}), 400
 
    main_grid_data = session['main_grid']
    top_grid_data = session['top_grid']
    click_history = session.get('click_history', [])
    grid_history = session.get("grid_history", [])
 
    # --- 新增：后端验证，防止在游戏结束后继续操作 ---
    # 如果棋盘已经满了，直接返回游戏结束状态，不执行任何操作。
    if 0 not in main_grid_data:
        print("Game is already over. Ignoring click.")
        return jsonify({
            'top_grid': top_grid_data,
            'main_grid': main_grid_data,
            'game_over': True
        })
    # --- 新增代码结束 ---
 
    game_over = False
 
    if clicked_index is not None and 0 <= clicked_index < 16:
        # 只有在格子为空时才执行逻辑
        if main_grid_data[clicked_index] == 0:
            click_history.append(clicked_index)
            grid_history.append(top_grid_data.copy())
            
            gm = Triple()
            gm.board = np.array(main_grid_data.copy())
            gm.num_pool = np.array(top_grid_data.copy())
            gm.place(clicked_index, gm.num_pool[0])
            gm.update_pool()
            main_grid_data = gm.board.copy().tolist()
            top_grid_data = gm.num_pool.copy().tolist()
 
            if 0 not in main_grid_data:
                game_over = True
                print("Game Over! Board is full.")
                print("score:", len(click_history))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SAVE_DIR, f"game_{timestamp}.json")
                game_data_to_save = {
                    'final_board': main_grid_data,
                    'click_history': click_history,
                    "grid_history": grid_history,
                    "score": len(click_history)  # 假设分数是点击次数
                }
                with open(filename, 'w') as f:
                    json.dump(game_data_to_save, f, indent=4)
                print(f"Game record saved to {filename}")
        else:
            # 如果点击的格子不为空，则不执行任何操作，直接返回当前状态
            print(f"Cell {clicked_index} is not empty. Ignoring.")
 
    session['top_grid'] = top_grid_data
    session['main_grid'] = main_grid_data
    session['click_history'] = click_history
    session['grid_history'] = grid_history
 
    response_data = {
        'top_grid': session['top_grid'],
        'main_grid': session['main_grid'],
        'game_over': game_over
    }
    return jsonify(response_data)
 
@app.route('/restart', methods=['POST'])
def restart_game():
    print("Restarting game for current session.")
    # 重置所有 session 数据
    session['top_grid'] = [1, 1]
    session['main_grid'] = np.zeros(16, dtype=int).tolist()
    session['click_history'] = [] # 同样重置历史记录
    session['grid_history'] = []
 
    response_data = {
        'top_grid': session['top_grid'],
        'main_grid': session['main_grid'],
        'game_over': False # 重启后游戏肯定没有结束
    }
    return jsonify(response_data)
 
if __name__ == '__main__':
    app.run(debug=True)