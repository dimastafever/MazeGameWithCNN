import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

pygame.init()
pygame.font.init()

CELL_SIZE = 50
GRID_W, GRID_H = 15, 15
MAZE_WIDTH = CELL_SIZE * GRID_W
MAZE_HEIGHT = CELL_SIZE * GRID_H
PANEL_WIDTH = 350
WIDTH = MAZE_WIDTH + PANEL_WIDTH
HEIGHT = MAZE_HEIGHT + 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Нейро-лабиринт: PPO Actor-Critic")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 18)
small_font = pygame.font.Font(None, 14)

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

class ActorCriticCNN(nn.Module):
    def __init__(self, grid_h, grid_w):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * grid_h * grid_w, 256)
        self.policy_head = nn.Linear(256, 4)
        self.value_head = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value

def state_to_tensor(agent_pos, goal_pos, walls):
    grid = np.zeros((3, GRID_H, GRID_W), dtype=np.float32)
    for y in range(GRID_H):
        for x in range(GRID_W):
            grid[0, y, x] = 1.0 if walls[y][x] else 0.0
    grid[1, agent_pos[1], agent_pos[0]] = 1.0
    grid[2, goal_pos[1], goal_pos[0]] = 1.0
    return torch.from_numpy(grid).unsqueeze(0)

def index_to_direction(idx):
    if idx == 0: return (0, -1)
    elif idx == 1: return (0, 1)
    elif idx == 2: return (-1, 0)
    else: return (1, 0)

# ---------- Функция для отрисовки графика ----------
def draw_line_graph(surface, x, y, w, h, data, color, label, y_min=None, y_max=None):
    pygame.draw.rect(surface, WHITE, (x, y, w, h), 1)
    if not data:
        return
    if y_min is None:
        y_min = min(data)
    if y_max is None:
        y_max = max(data)
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0
    margin = (y_max - y_min) * 0.05
    y_min -= margin
    y_max += margin

    n = len(data)
    step_x = w / max(1, n - 1)
    points = []
    for i, val in enumerate(data):
        px = x + i * step_x
        py = y + h - (val - y_min) / (y_max - y_min) * h
        points.append((px, py))
    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, 2)
    label_surf = small_font.render(label, True, color)
    surface.blit(label_surf, (x + 5, y + 2))

walls = [[False] * GRID_W for _ in range(GRID_H)]
for x in range(GRID_W):
    walls[0][x] = True
    walls[GRID_H-1][x] = True
for y in range(GRID_H):
    walls[y][0] = True
    walls[y][GRID_W-1] = True

start_pos = (1, 1)
goal_pos = (GRID_W-2, GRID_H-2)
agent_pos = start_pos

net = ActorCriticCNN(GRID_H, GRID_W)
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# Гиперпараметры PPO
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999
gamma = 0.99
max_steps_per_episode = 800
entropy_coeff = 0.01
clip_epsilon = 0.2
ppo_epochs = 4

training_mode = False
fast_training = False

episode_count = 0
step_count = 0
total_reward_episode = 0
last_100_rewards = deque(maxlen=100)
last_100_success = deque(maxlen=100)
visited_cells = set()

# Буфер для текущего эпизода
episode_states = []
episode_actions = []
episode_rewards = []
episode_log_probs = []

history_rewards = []
history_success = []
history_epsilon = []
history_loss = []

# Лучшие результаты для итогового вывода
best_avg_reward = -float('inf')
best_success_rate = 0.0
best_episode = 0

running = True
last_step_time = time.time()
step_interval = 0.02

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def ppo_update(states, actions, old_log_probs, returns, net, optimizer, clip_epsilon, entropy_coeff, ppo_epochs):
    states_tensor = torch.cat(states)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)
    returns_tensor = returns

    for _ in range(ppo_epochs):
        policy, values = net(states_tensor)
        values = values.squeeze(1)
        log_probs = torch.log(policy.gather(1, actions_tensor.unsqueeze(1)).squeeze(1))

        advantages = returns_tensor - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.functional.mse_loss(values, returns_tensor)
        entropy = -(policy * torch.log(policy + 1e-10)).sum(dim=-1).mean()
        loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()

def reset_episode():
    global agent_pos, step_count, total_reward_episode, visited_cells, episode_states, episode_actions, episode_rewards, episode_log_probs
    agent_pos = start_pos
    step_count = 0
    total_reward_episode = 0
    visited_cells = {agent_pos}
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_log_probs = []

def print_summary():
    """Вывод итоговых результатов обучения в консоль."""
    avg_reward = np.mean(last_100_rewards) if last_100_rewards else 0
    avg_success = np.mean(last_100_success) * 100 if last_100_success else 0
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("="*50)
    print(f"Всего эпизодов: {episode_count}")
    print(f"Средняя награда (последние 100): {avg_reward:.2f}")
    print(f"Успешность (последние 100): {avg_success:.1f}%")
    print(f"Лучшая средняя награда: {best_avg_reward:.2f} (эпизод {best_episode})")
    print(f"Лучшая успешность: {best_success_rate:.1f}%")
    print(f"Текущий epsilon: {epsilon:.3f}")
    print("="*50)

def draw_screen():
    screen.fill(BLACK)
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if walls[y][x]:
                pygame.draw.rect(screen, RED, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect, 1)
    pygame.draw.rect(screen, YELLOW, (start_pos[0]*CELL_SIZE, start_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, GREEN, (goal_pos[0]*CELL_SIZE, goal_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    center = (agent_pos[0]*CELL_SIZE + CELL_SIZE//2, agent_pos[1]*CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, BLUE, center, CELL_SIZE//3)

    panel_x = MAZE_WIDTH + 10
    info_lines = [
        f"Режим: {'Обучение' if training_mode else 'Редактор'}{' (быстро)' if fast_training else ''}",
        f"Эпизодов: {episode_count}",
        f"Шагов: {step_count}/{max_steps_per_episode}",
        f"Награда тек.: {total_reward_episode:.2f}",
        f"Epsilon: {epsilon:.3f}",
        f"Ср. награда (100): {np.mean(last_100_rewards) if last_100_rewards else 0:.2f}",
        f"Успешность (100): {np.mean(last_100_success)*100 if last_100_success else 0:.1f}%",
    ]
    for i, text in enumerate(info_lines):
        screen.blit(font.render(text, True, YELLOW), (panel_x, 5 + i*20))

    graph_x = panel_x
    graph_w = PANEL_WIDTH - 20
    graph_h = 80
    y_pos = 160

    draw_line_graph(screen, graph_x, y_pos, graph_w, graph_h, history_rewards[-200:], BLUE, "Reward")
    y_pos += graph_h + 10
    draw_line_graph(screen, graph_x, y_pos, graph_w, graph_h, history_success[-200:], GREEN, "Success %", y_min=0, y_max=100)
    y_pos += graph_h + 10
    draw_line_graph(screen, graph_x, y_pos, graph_w, graph_h, history_epsilon[-200:], ORANGE, "Epsilon", y_min=0, y_max=1.0)
    y_pos += graph_h + 10
    draw_line_graph(screen, graph_x, y_pos, graph_w, graph_h, history_loss[-200:], RED, "Loss")

    help_text = "ЛКМ - стена, ПКМ - старт, колесо - финиш, ПРОБЕЛ - обучение, F - быстрое обучение, R - сброс сети, C - очистить, ESC - выход"
    screen.blit(font.render(help_text, True, WHITE), (10, MAZE_HEIGHT + 10))
    pygame.display.flip()

def process_events():
    global running, training_mode, fast_training, start_pos, goal_pos, agent_pos, walls, epsilon, episode_count, history_rewards, history_success, history_epsilon, history_loss, last_100_rewards, last_100_success
    global net, optimizer, best_avg_reward, best_success_rate, best_episode
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print_summary()
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                training_mode = not training_mode
                if training_mode:
                    print("Обучение запущено")
                    reset_episode()
                else:
                    print("Обучение остановлено")
                    print_summary()
            elif event.key == pygame.K_f:
                fast_training = not fast_training
                print(f"Быстрое обучение: {'включено' if fast_training else 'выключено'}")
            elif event.key == pygame.K_r:
                net = ActorCriticCNN(GRID_H, GRID_W)
                optimizer = optim.Adam(net.parameters(), lr=0.0003)
                episode_count = 0
                epsilon = 1.0
                last_100_rewards.clear()
                last_100_success.clear()
                history_rewards.clear()
                history_success.clear()
                history_epsilon.clear()
                history_loss.clear()
                best_avg_reward = -float('inf')
                best_success_rate = 0.0
                best_episode = 0
                print("Модель сброшена")
            elif event.key == pygame.K_c:
                for y in range(1, GRID_H-1):
                    for x in range(1, GRID_W-1):
                        walls[y][x] = False
                print("Лабиринт очищен")
            elif event.key == pygame.K_ESCAPE:
                print_summary()
                running = False

        if not training_mode:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx < MAZE_WIDTH:
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        if event.button == 1:
                            walls[gy][gx] = not walls[gy][gx]
                        elif event.button == 3:
                            if not walls[gy][gx]:
                                start_pos = (gx, gy)
                                agent_pos = start_pos
                        elif event.button == 2:
                            if not walls[gy][gx]:
                                goal_pos = (gx, gy)
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:
                    mx, my = pygame.mouse.get_pos()
                    if mx < MAZE_WIDTH:
                        gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                            if pygame.mouse.get_pressed()[0]:
                                walls[gy][gx] = True
                            elif pygame.mouse.get_pressed()[2]:
                                walls[gy][gx] = False

while running:
    process_events()

    if training_mode:
        if fast_training:
            batch_time = 0.1
            start_batch = time.time()
            while time.time() - start_batch < batch_time:
                if not training_mode or not fast_training:
                    break
                state_tensor = state_to_tensor(agent_pos, goal_pos, walls)
                policy, _ = net(state_tensor)
                policy = policy.detach().squeeze(0).numpy()

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    action = int(np.argmax(policy))

                log_prob = np.log(policy[action] + 1e-10)
                episode_log_probs.append(log_prob)

                episode_states.append(state_tensor)
                episode_actions.append(action)

                dx, dy = index_to_direction(action)
                new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
                moved = False
                if 0 <= new_x < GRID_W and 0 <= new_y < GRID_H and not walls[new_y][new_x]:
                    agent_pos = (new_x, new_y)
                    moved = True

                reward = -0.005
                if agent_pos == goal_pos:
                    reward = 2.0
                elif not moved:
                    reward = -0.05
                else:
                    old_dist = manhattan_distance((agent_pos[0]-dx, agent_pos[1]-dy), goal_pos)
                    new_dist = manhattan_distance(agent_pos, goal_pos)
                    reward += (old_dist - new_dist) * 0.05
                    if agent_pos in visited_cells:
                        reward -= 0.01
                    else:
                        visited_cells.add(agent_pos)
                        reward += 0.005

                episode_rewards.append(reward)
                total_reward_episode += reward
                step_count += 1

                done = agent_pos == goal_pos or step_count >= max_steps_per_episode

                if done:
                    returns = compute_returns(episode_rewards, gamma)
                    loss_val = ppo_update(episode_states, episode_actions, episode_log_probs, returns, net, optimizer, clip_epsilon, entropy_coeff, ppo_epochs)
                    history_loss.append(loss_val)

                    episode_count += 1
                    success = 1 if agent_pos == goal_pos else 0
                    last_100_rewards.append(total_reward_episode)
                    last_100_success.append(success)

                    avg_reward = np.mean(last_100_rewards) if last_100_rewards else 0
                    avg_success = np.mean(last_100_success) * 100 if last_100_success else 0

                    # Обновляем лучшие результаты
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_episode = episode_count
                    if avg_success > best_success_rate:
                        best_success_rate = avg_success

                    history_rewards.append(avg_reward)
                    history_success.append(avg_success)
                    history_epsilon.append(epsilon)

                    if epsilon > epsilon_min:
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)

                    if episode_count % 10 == 0:
                        print(f"Эпизод {episode_count}, средняя награда (100): {avg_reward:.2f}, успешность: {avg_success:.1f}%, epsilon: {epsilon:.3f}")

                    # Автоматическая остановка при достижении порога
                    if len(last_100_success) == 100 and avg_success >= 95.0:
                        print("Достигнута высокая успешность! Обучение остановлено.")
                        print_summary()
                        training_mode = False
                        reset_episode()
                        break

                    reset_episode()

            draw_screen()
            clock.tick(60)
        else:
            if time.time() - last_step_time > step_interval:
                last_step_time = time.time()

                state_tensor = state_to_tensor(agent_pos, goal_pos, walls)
                policy, _ = net(state_tensor)
                policy = policy.detach().squeeze(0).numpy()

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    action = int(np.argmax(policy))

                log_prob = np.log(policy[action] + 1e-10)
                episode_log_probs.append(log_prob)

                episode_states.append(state_tensor)
                episode_actions.append(action)

                dx, dy = index_to_direction(action)
                new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
                moved = False
                if 0 <= new_x < GRID_W and 0 <= new_y < GRID_H and not walls[new_y][new_x]:
                    agent_pos = (new_x, new_y)
                    moved = True

                reward = -0.005
                if agent_pos == goal_pos:
                    reward = 2.0
                elif not moved:
                    reward = -0.05
                else:
                    old_dist = manhattan_distance((agent_pos[0]-dx, agent_pos[1]-dy), goal_pos)
                    new_dist = manhattan_distance(agent_pos, goal_pos)
                    reward += (old_dist - new_dist) * 0.05
                    if agent_pos in visited_cells:
                        reward -= 0.01
                    else:
                        visited_cells.add(agent_pos)
                        reward += 0.005

                episode_rewards.append(reward)
                total_reward_episode += reward
                step_count += 1

                done = agent_pos == goal_pos or step_count >= max_steps_per_episode

                if done:
                    returns = compute_returns(episode_rewards, gamma)
                    loss_val = ppo_update(episode_states, episode_actions, episode_log_probs, returns, net, optimizer, clip_epsilon, entropy_coeff, ppo_epochs)
                    history_loss.append(loss_val)

                    episode_count += 1
                    success = 1 if agent_pos == goal_pos else 0
                    last_100_rewards.append(total_reward_episode)
                    last_100_success.append(success)

                    avg_reward = np.mean(last_100_rewards) if last_100_rewards else 0
                    avg_success = np.mean(last_100_success) * 100 if last_100_success else 0

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_episode = episode_count
                    if avg_success > best_success_rate:
                        best_success_rate = avg_success

                    history_rewards.append(avg_reward)
                    history_success.append(avg_success)
                    history_epsilon.append(epsilon)

                    if epsilon > epsilon_min:
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)

                    if len(last_100_success) == 100 and avg_success >= 95.0:
                        print("Достигнута высокая успешность! Обучение остановлено.")
                        print_summary()
                        training_mode = False
                        reset_episode()
                        break

                    reset_episode()

    if not fast_training:
        draw_screen()
        clock.tick(60)
    else:
        if not training_mode:
            draw_screen()
            clock.tick(60)

pygame.quit()