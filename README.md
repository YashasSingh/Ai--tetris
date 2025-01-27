# Tetris AI: Smarter Gameplay Through Machine Learning

An advanced AI that masters [Tetris](https://en.wikipedia.org/wiki/Tetris) using deep reinforcement learning techniques.  

## Overview

This project leverages deep Q-learning to create an agent that trains itself to excel at Tetris. Starting from random moves, it progressively learns to make intelligent decisions, eventually achieving high scores with optimized gameplay strategies.  


---

## Prerequisites  

To replicate or modify this project, you'll need:  
- A deep learning framework (TensorFlow, Jax, or PyTorch)  
- TensorBoard for visualization  
- Keras for building neural networks  
- OpenCV-Python for visual processing  
- NumPy for numerical operations  
- Pillow for image processing  
- Tqdm for progress bars  

Note: Jax (`jax[cuda12]`) with Keras is recommended for faster computation (`KERAS_BACKEND="jax" python3 run.py`).

---

## Usage  

### Training the AI
Run the script to start training. Customize hyperparameters in the `run.py` file.  
```bash
python3 run.py
```

### Viewing Training Logs
Monitor progress with TensorBoard:  
```bash
tensorboard --logdir ./logs
```

### Playing Tetris with a Trained Model
Use a pre-trained model (`sample.keras`) to see the AI in action:  
```bash
python3 run_model.py sample.keras
```

---

## How It Works  

### Learning Through Reinforcement  

The AI begins by making random moves. Over time, it uses reinforcement learning principles to train itself:  
1. **Replay Memory:** Stores game states and rewards in a queue.  
2. **Q-Learning:** Evaluates actions not just by immediate rewards but also potential future rewards.  
3. **Exploration-Exploitation:** Balances exploring new moves with sticking to known strategies for better outcomes.  

### Training with Q-Learning  

The AI uses the Q-Learning algorithm to maximize long-term rewards:  
- It predicts the best possible score for future states.  
- Neural network updates consider both immediate and discounted future rewards.  
- This approach avoids greediness, enabling the agent to take moves with delayed benefits.  

### Deciding the Best Move  

Instead of relying on a single-action prediction, the AI evaluates all possible moves and predicts the resulting scores. The move with the highest predicted score is executed.  

### Game State Features  

The following attributes are used for training the model (top four proved most essential):  
- Lines cleared  
- Number of holes  
- Bumpiness (height difference between adjacent columns)  
- Total height  

---

## Technical Details  

### Neural Network Design  
The agent's neural network uses:  
- 2 hidden layers with 32 neurons each  
- ReLU activation for hidden layers, Linear for output layer  
- Mean Squared Error for loss  
- Adam optimizer  
- Exploration rate (`epsilon`) gradually decreasing from 1 to 0  

### Training Parameters  
- Replay queue size: 20,000  
- Batch size: 512  
- Discount factor: 0.95  
- Episodes: 2,000  

---

