import os
import threading
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file, Response

from agent import Agent
from web_game import WebSnakeGame, Direction, AI_SPEED, BASE_SPEED

# Signal to suppress interactive plotting in agent/helper.
os.environ["SNAKE_WEB"] = "1"

app = Flask(__name__)


class ServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "ai"  # "ai" or "human"
        self.direction = Direction.RIGHT
        self.last_frame = b""
        self.scores = []
        self.mean_scores = []
        self.total_score = 0
        self.record = 0
        self.last_q_values = [0.0, 0.0, 0.0]
        self.last_action = [1, 0, 0]
        self.last_scores_chart = b""
        self.last_q_chart = b""
        self.game = WebSnakeGame()
        self.agent = Agent()


state = ServerState()


def _render_scores_chart(scores, mean_scores):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if scores:
        ax.plot(scores, label="Score")
        ax.plot(mean_scores, label="Mean Score")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
        ax.set_title("Training Progress")
        ax.set_xlabel("Games")
        ax.set_ylabel("Score")
    else:
        ax.text(0.5, 0.5, "No games yet", ha="center", va="center")
        ax.set_axis_off()

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="PNG")
    plt.close(fig)
    return buffer.getvalue()


def _render_q_chart(q_values, action):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    actions = ["Straight", "Turn Right", "Turn Left"]
    colors = ["#2ecc71" if i == int(action.index(1)) else "#3498db" for i in range(3)]
    bars = ax.bar(actions, q_values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, q_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.05 if val >= 0 else -0.15),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Q-Values (Live)")
    ax.set_ylabel("Q-Value")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="PNG")
    plt.close(fig)
    return buffer.getvalue()


def training_loop():
    last_chart_time = 0.0
    while True:
        with state.lock:
            mode = state.mode
            direction = state.direction

        if mode == "ai":
            game = state.game
            agent = state.agent

            state_old = agent.get_state(game)
            action = agent.get_action(state_old, game)
            game.last_state = state_old
            game.last_action = action

            reward, done, score = game.step_ai(action)
            game.last_reward = reward
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > state.record:
                    state.record = score
                    agent.model.save()

                state.scores.append(score)
                state.total_score += score
                mean_score = state.total_score / agent.n_games
                state.mean_scores.append(mean_score)
        else:
            game = state.game
            done, _score = game.step_human(direction)
            if done:
                game.reset()

        frame_bytes = state.game.get_frame_bytes()

        now = time.time()
        if now - last_chart_time > 0.5:
            last_chart_time = now
            state.last_scores_chart = _render_scores_chart(state.scores, state.mean_scores)
            state.last_q_chart = _render_q_chart(state.game.last_q_values, state.game.last_action or [1, 0, 0])

        with state.lock:
            state.last_frame = frame_bytes
            state.last_q_values = state.game.last_q_values
            state.last_action = state.game.last_action or [1, 0, 0]

        fps = AI_SPEED if mode == "ai" else BASE_SPEED
        time.sleep(max(0.001, 1.0 / fps))


@app.route("/")
def index():
    return Response(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Snake RL</title>
  <style>
    body { font-family: Georgia, serif; background: #111; color: #eee; margin: 0; }
    .wrap { display: grid; grid-template-columns: 1fr 420px; gap: 16px; padding: 16px; }
    .panel { background: #1a1a1a; padding: 12px; border-radius: 8px; }
    .controls button { margin: 4px; padding: 8px 12px; background: #333; color: #fff; border: 0; border-radius: 6px; cursor: pointer; }
    .controls button.active { background: #2ecc71; }
    img { width: 100%; height: auto; display: block; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <div class="controls">
        <button id="mode-ai" class="active">AI Mode</button>
        <button id="mode-human">Human Mode</button>
      </div>
      <img id="game" alt="game" />
      <div class="controls">
        <button data-dir="up">Up</button>
        <button data-dir="left">Left</button>
        <button data-dir="down">Down</button>
        <button data-dir="right">Right</button>
      </div>
      <p>Use arrow keys or buttons in Human mode.</p>
    </div>
    <div class="panel">
      <div class="grid">
        <img id="scores" alt="scores" />
        <img id="qvalues" alt="qvalues" />
      </div>
    </div>
  </div>

<script>
  const gameImg = document.getElementById('game');
  const scoresImg = document.getElementById('scores');
  const qImg = document.getElementById('qvalues');
  const modeAi = document.getElementById('mode-ai');
  const modeHuman = document.getElementById('mode-human');

  function refresh() {
    const t = Date.now();
    gameImg.src = `/frame?t=${t}`;
    scoresImg.src = `/chart/scores?t=${t}`;
    qImg.src = `/chart/qvalues?t=${t}`;
  }
  setInterval(refresh, 100);
  refresh();

  async function setMode(mode) {
    await fetch('/mode', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ mode }) });
    modeAi.classList.toggle('active', mode === 'ai');
    modeHuman.classList.toggle('active', mode === 'human');
  }
  modeAi.addEventListener('click', () => setMode('ai'));
  modeHuman.addEventListener('click', () => setMode('human'));

  async function sendDir(dir) {
    await fetch('/action', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ direction: dir }) });
  }

  document.querySelectorAll('[data-dir]').forEach(btn => {
    btn.addEventListener('click', () => sendDir(btn.dataset.dir));
  });

  window.addEventListener('keydown', (e) => {
    const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right' };
    if (map[e.key]) {
      sendDir(map[e.key]);
    }
  });
</script>
</body>
</html>
""",
        mimetype="text/html",
    )


@app.route("/frame")
def frame():
    with state.lock:
        data = state.last_frame
    return send_file(BytesIO(data), mimetype="image/png")


@app.route("/chart/scores")
def chart_scores():
    with state.lock:
        data = state.last_scores_chart
    return send_file(BytesIO(data), mimetype="image/png")


@app.route("/chart/qvalues")
def chart_qvalues():
    with state.lock:
        data = state.last_q_chart
    return send_file(BytesIO(data), mimetype="image/png")


@app.route("/mode", methods=["POST"])
def mode():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "ai")
    if mode not in ("ai", "human"):
        return jsonify({"ok": False, "error": "invalid mode"}), 400

    with state.lock:
        state.mode = mode
        state.game.reset()
    return jsonify({"ok": True, "mode": mode})


@app.route("/action", methods=["POST"])
def action():
    payload = request.get_json(silent=True) or {}
    direction = payload.get("direction", "")
    mapping = {
        "up": Direction.UP,
        "down": Direction.DOWN,
        "left": Direction.LEFT,
        "right": Direction.RIGHT,
    }
    if direction not in mapping:
        return jsonify({"ok": False, "error": "invalid direction"}), 400

    with state.lock:
        state.direction = mapping[direction]
    return jsonify({"ok": True})


if __name__ == "__main__":
    loop = threading.Thread(target=training_loop, daemon=True)
    loop.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
