import torch
import matplotlib.pyplot as plt

def load_model_and_metrics(file_path):
    checkpoint = torch.load(file_path)
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    metrics = checkpoint['metrics']
    return model_state_dict, optimizer_state_dict, metrics

# Assuming the path to your saved model
file_path = '/Users/tejaswinibharatha/Desktop/project/cups-rl/cups-rl/algorithms/a3c/saved_models/model_3.pth'
model_state_dict, optimizer_state_dict, metrics = load_model_and_metrics(file_path)

episode_rewards = metrics['episode_rewards']

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, marker='o', linestyle='-', color='b')
plt.title("Rewards vs Episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.grid(True)
plt.show()
