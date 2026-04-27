import matplotlib

try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

_backend_name = plt.get_backend().lower()
_interactive_backend = _backend_name != "agg"

try:
    from IPython import display  # type: ignore
    _ipython_available = True
except ImportError:  # pragma: no cover - optional dependency
    display = None
    _ipython_available = False

plt.ion()

train_fig, train_ax = None, None
edu_fig, edu_axes = None, None
_edu_last_episode = None
_edu_steps_this_episode = 0

STATE_LABELS = [
    "danger straight", "danger right", "danger left",
    "moving L", "moving R", "moving U", "moving D",
    "food L", "food R", "food U", "food D"
]


def _place_window(fig, width, height, x_ratio, y_ratio):
    try:
        window = fig.canvas.manager.window
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = int(max(0, min(screen_w - width, screen_w * x_ratio)))
        y = int(max(0, min(screen_h - height, screen_h * y_ratio)))
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.deiconify()
        window.lift()
    except Exception:
        pass


def _refresh_figure():
    if _ipython_available:
        try:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            return
        except Exception:
            # fall back to non-IPython rendering if notebook display fails
            pass
    try:
        plt.draw()
    except Exception:
        # If the matplotlib backend (e.g. Tkinter) is shutting down or
        # has a race condition removing callbacks, ignore errors so the
        # training loop can continue or exit cleanly.
        return


def plot(scores, mean_scores):
    global train_fig, train_ax

    if not scores or not mean_scores:
        return

    if train_fig is None:
        train_fig, train_ax = plt.subplots(figsize=(9, 6), dpi=120)
        train_fig.canvas.manager.set_window_title('Training Progress')
        _place_window(train_fig, 980, 650, 0.52, 0.04)

        train_ax.clear()
        train_ax.set_title('Training Progress')
        train_ax.text(0.5, 0.5, 'Waiting for first completed game...', ha='center', va='center')
        train_ax.set_axis_off()

        try:
            if _interactive_backend:
                train_fig.show()
        except Exception:
            pass

    train_ax.clear()
    train_ax.set_title('Training...')
    train_ax.set_xlabel('Number of Games')
    train_ax.set_ylabel('Score')
    train_ax.plot(scores, label='Score')
    train_ax.plot(mean_scores, label='Mean Score')
    train_ax.legend(loc='upper left')
    train_ax.set_ylim(ymin=0)
    train_ax.text(len(scores) - 1, scores[-1], str(scores[-1]))
    train_ax.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    try:
        if not _interactive_backend:
            return
        train_fig.canvas.draw_idle()
        train_fig.canvas.flush_events()
        plt.pause(0.03)
    except Exception:
        pass


# Q-value live bar chart window
q_fig, q_ax = None, None


def plot_q_values(q_values, action_taken, frame, game_number):
    global q_fig, q_ax

    if q_fig is None:
        q_fig, q_ax = plt.subplots(figsize=(8, 6), dpi=120)
        q_fig.canvas.manager.set_window_title('Q-Values - Live')
        _place_window(q_fig, 900, 650, 0.02, 0.04)

        q_ax.clear()
        q_ax.set_title('Q-Values - Live')
        q_ax.text(0.5, 0.5, 'Waiting for Q-value updates...', ha='center', va='center', color='white')
        q_ax.set_axis_off()
        q_fig.patch.set_facecolor('#16213e')

        try:
            if _interactive_backend:
                q_fig.show()
        except Exception:
            pass

    q_ax.clear()

    actions = ['Straight', 'Turn Right', 'Turn Left']
    colors = ['#2ecc71' if i == action_taken else '#3498db' for i in range(3)]

    bars = q_ax.bar(actions, q_values, color=colors, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, q_values):
        q_ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.05 if val >= 0 else -0.15),
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=11, color='white'
        )

    q_ax.set_title(f'Q-Values  |  Game {game_number}  |  Frame {frame}', fontsize=11)
    q_ax.set_ylabel('Q-Value (expected reward)')
    q_ax.set_facecolor('#1a1a2e')
    q_fig.patch.set_facecolor('#16213e')
    q_ax.tick_params(colors='white')
    q_ax.yaxis.label.set_color('white')
    q_ax.title.set_color('white')
    for spine in q_ax.spines.values():
        spine.set_edgecolor('#444')

    q_ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#2ecc71', label='Chosen action'),
            plt.Rectangle((0, 0), 1, 1, color='#3498db', label='Other actions')
        ],
        loc='upper right', fontsize=9,
        facecolor='#1a1a2e', labelcolor='white', edgecolor='#444'
    )

    try:
        if not _interactive_backend:
            return
        q_fig.canvas.draw_idle()
        q_fig.canvas.flush_events()
        plt.pause(0.03)
    except Exception:
        pass


def plot_educational(state, reward, n_games, epsilon, memory_size, total_steps):
    global edu_fig, edu_axes, _edu_last_episode, _edu_steps_this_episode

    if state is None:
        return

    if _edu_last_episode != n_games:
        _edu_last_episode = n_games
        _edu_steps_this_episode = 1
    else:
        _edu_steps_this_episode += 1

    s = [int(v) for v in state]
    danger_info = []
    if s[0]:
        danger_info.append("straight")
    if s[1]:
        danger_info.append("right")
    if s[2]:
        danger_info.append("left")

    if epsilon > 20:
        behavior = f"Agent is exploring randomly because epsilon ({epsilon:.2f}) is still high."
    else:
        behavior = f"Agent exploiting learned Q-values as epsilon ({epsilon:.2f}) is lower."

    if danger_info:
        behavior += " Danger detected to the " + ", ".join(danger_info) + "."
    else:
        behavior += " No immediate danger detected around the next move."

    if edu_fig is None:
        edu_fig, edu_axes = plt.subplots(2, 2, figsize=(10, 7), dpi=120)
        edu_fig.canvas.manager.set_window_title("RL Education — What the Agent is Thinking")
        _place_window(edu_fig, 980, 700, 0.26, 0.08)
        try:
            if _interactive_backend:
                edu_fig.show()
        except Exception:
            pass

    ax_state, ax_info, ax_eps, ax_mem = edu_axes.flatten()

    # 1) State vector bits as colored squares.
    ax_state.clear()
    state_grid = [s]
    ax_state.imshow(state_grid, cmap=plt.cm.colors.ListedColormap(["#8a8a8a", "#2ecc71"]), aspect="auto", vmin=0, vmax=1)
    ax_state.set_title("State Vector (11 Bits)")
    ax_state.set_yticks([])
    ax_state.set_xticks(range(len(STATE_LABELS)))
    ax_state.set_xticklabels(STATE_LABELS, rotation=35, ha="right", fontsize=8)
    for idx, bit in enumerate(s):
        ax_state.text(idx, 0, str(bit), ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    # 2) Current step info and plain-English explanation.
    ax_info.clear()
    ax_info.set_title("Current Step Info")
    ax_info.axis("off")
    step_text = (
        f"Current reward: {reward:+.0f}\n"
        f"Episode number: {n_games}\n"
        f"Steps this episode: {_edu_steps_this_episode}\n"
        f"Total steps trained: {total_steps}\n\n"
        f"What is happening:\n{behavior}"
    )
    ax_info.text(0.02, 0.98, step_text, va="top", ha="left", fontsize=10, wrap=True)

    # 3) Epsilon decay curve from game 0 to current game.
    ax_eps.clear()
    eps_x = list(range(n_games + 1))
    eps_y = [max(0, 80 - g) for g in eps_x]
    ax_eps.plot(eps_x, eps_y, color="#1f77b4", linewidth=2, label="epsilon = 80 - n_games")
    ax_eps.scatter([n_games], [max(0, epsilon)], color="#e74c3c", zorder=3, label=f"current: {epsilon:.2f}")
    ax_eps.set_title("Epsilon Decay Curve")
    ax_eps.set_xlabel("Game")
    ax_eps.set_ylabel("Epsilon")
    ax_eps.set_ylim(bottom=0)
    ax_eps.grid(alpha=0.3)
    ax_eps.legend(loc="upper right", fontsize=8)

    # 4) Replay memory and training stats.
    ax_mem.clear()
    ax_mem.set_title("Memory & Training Stats")
    ax_mem.axis("off")
    mem_text = (
        f"Replay buffer size: {memory_size}\n"
        f"Total steps trained: {total_steps}\n"
        f"Batch size: 1000\n\n"
        "Experience replay stores past transitions and trains on random mini-batches\n"
        "to reduce correlation between consecutive experiences."
    )
    ax_mem.text(0.02, 0.98, mem_text, va="top", ha="left", fontsize=10, wrap=True)

    try:
        if not _interactive_backend:
            return
        edu_fig.tight_layout()
        edu_fig.canvas.draw()
        edu_fig.canvas.flush_events()
        plt.pause(0.03)
    except Exception:
        pass
