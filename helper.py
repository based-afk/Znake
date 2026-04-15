import matplotlib.pyplot as plt

try:
    from IPython import display  # type: ignore
    _ipython_available = True
except ImportError:  # pragma: no cover - optional dependency
    display = None
    _ipython_available = False

plt.ion()

train_fig, train_ax = None, None


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
        train_fig, train_ax = plt.subplots(figsize=(6, 4))
        train_fig.canvas.manager.set_window_title('Training Progress')
        try:
            train_fig.canvas.manager.window.wm_geometry('+980+50')
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
        train_fig.canvas.draw()
        train_fig.canvas.flush_events()
        plt.pause(0.1)
    except Exception:
        pass


# Q-value live bar chart window
q_fig, q_ax = None, None


def plot_q_values(q_values, action_taken, frame, game_number):
    global q_fig, q_ax

    if q_fig is None:
        q_fig, q_ax = plt.subplots(figsize=(5, 4))
        q_fig.canvas.manager.set_window_title('Q-Values - Live')
        try:
            q_fig.canvas.manager.window.wm_geometry('+1530+50')
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
        q_fig.canvas.draw()
        q_fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass
