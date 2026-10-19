import csv
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # lets pygame render without a window

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# SETTINGS
# ============================================================
SUMMARY_FILE = "results/experiment_summary.csv"
CONFIG_OVERRIDE = None      # e.g. "alpha_0.5" to force a specific agent; None = pick the best one
N_GAMES = 10                # how many hands to show
SEED = 7                    # which deals to show
FPS = 1.5                   # frames per second (lower = slower)
OUTPUT = "results/blackjack_agent.mp4"
# ============================================================

SCALE = 2                   # enlarge the game picture 2x
BANNER = 144                # height of the caption bar under the picture

try:
    FONT = ImageFont.load_default(size=30)
    SMALL = ImageFont.load_default(size=24)
except TypeError:           # very old Pillow: tiny default font
    FONT = SMALL = ImageFont.load_default()


def hand_total(hand):
    """Blackjack total of a list of cards (an ace counts 11 if that does not bust)."""
    total = sum(hand)
    return total + 10 if 1 in hand and total + 10 <= 21 else total


def make_frame(game_frame, line1, line2, color):
    """Enlarge the rendered table and add a caption bar underneath."""
    img = Image.fromarray(game_frame)
    w, h = img.size
    img = img.resize((w * SCALE, h * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + BANNER), color)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, img.height + 18), line1, fill="white", font=FONT)
    draw.text((20, img.height + 78), line2, fill="white", font=SMALL)
    return np.array(canvas)


NEUTRAL, WIN_C, LOSS_C, DRAW_C = (40, 44, 60), (30, 120, 60), (150, 40, 40), (90, 90, 90)

# ---- choose which trained agent to show ----
with open(SUMMARY_FILE) as f:
    rows = list(csv.DictReader(f))

if CONFIG_OVERRIDE:
    name = CONFIG_OVERRIDE
else:
    best = max(rows, key=lambda r: float(r["mean_reward"]))
    name = best["configuration"]
    print(f"Best configuration: {name} (mean reward {float(best['mean_reward']):+.4f})")

Q = np.load(f"results/{name}_Q.npy")

# ---- play greedy games and capture captioned frames ----
env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode="rgb_array")
frames = []
wins = draws = losses = 0

for game in range(N_GAMES):
    obs, _ = env.reset(seed=SEED) if game == 0 else env.reset()
    record = f"Record so far: {wins} W  {draws} D  {losses} L"
    title = f"Hand {game + 1} of {N_GAMES}"
    ace = " (usable ace)" if obs[2] else ""
    frames += [make_frame(env.render(), title,
                          f"You: {obs[0]}{ace}   Dealer shows: {obs[1]}   |   {record}",
                          NEUTRAL)] * 2
    done = False

    while not done:
        action = int(np.argmax(Q[int(obs[0]), int(obs[1]), int(obs[2])]))   # greedy: no exploration
        move = "HIT" if action == 1 else "STICK"
        q = Q[int(obs[0]), int(obs[1]), int(obs[2])]
        frames += [make_frame(env.render(), f"{title}: agent decides {move}",
                              f"Q(stick) = {q[0]:+.3f}   Q(hit) = {q[1]:+.3f}   (picks the higher)",
                              NEUTRAL)] * 2
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if not done:
            frames += [make_frame(env.render(), f"{title}: agent chose {move}",
                                  f"You now have {obs[0]}   Dealer shows: {obs[1]}", NEUTRAL)] * 2
        else:
            player = dealer = None
            try:
                player = hand_total(env.unwrapped.player)
                dealer = hand_total(env.unwrapped.dealer)
            except AttributeError:
                pass
            if reward > 0:
                wins += 1
                word, color = "WIN", WIN_C
            elif reward < 0:
                losses += 1
                word, color = "LOSS", LOSS_C
            else:
                draws += 1
                word, color = "DRAW", DRAW_C
            if player is not None and player > 21:
                detail = f"Agent chose {move} and went over: {player} (BUST)"
            elif player is not None:
                detail = f"Agent chose {move}.  You: {player}   Dealer: {dealer}"
            else:
                detail = f"Agent chose {move}."
            frames += [make_frame(env.render(), f"{word}  (reward {reward:+.0f})", detail, color)] * 4
    print(f"Hand {game + 1}: {word}")

env.close()

# ---- write the MP4 ----
os.makedirs("results", exist_ok=True)
imageio.mimwrite(OUTPUT, frames, fps=FPS)
print(f"\nSaved {OUTPUT}: {len(frames)} frames, about {len(frames) / FPS:.0f} seconds")
print(f"Final record: {wins} wins, {draws} draws, {losses} losses")