import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

# Constants
ROWS, COLS = 6, 7

# Create Model
def create_model():
    """
    Creates a simple feedforward neural network (MLP) to predict AI moves.
    - Input: Board state flattened (6 rows x 7 columns).
    - Hidden layers: Two dense layers with 64 neurons and ReLU activation.
    - Output layer: 7 neurons (one per column) with softmax activation for probabilities.
    
    Why these layers?
    - `Dense(64, activation='relu')`: Adds 64 neurons with ReLU (Rectified Linear Unit) activation, which helps the model learn non-linear relationships.
    - `Dense(7, activation='softmax')`: Outputs probabilities for each column so the AI can "choose" a column.

    Why Adam optimizer?
    - Adam (Adaptive Moment Estimation) combines two optimization techniques (momentum and RMSProp) to adapt learning rates for each parameter. This speeds up training and reduces the chance of getting stuck in local minima.

    Why categorical_crossentropy loss?
    - Categorical Crossentropy compares the predicted probabilities with the true column (target) and measures how well the model performs. This loss function is commonly used in classification tasks.

    Why accuracy?
    - Accuracy calculates the percentage of predictions the model gets correct, helping us measure performance.
    """
    model = Sequential([
        Input(shape=(ROWS * COLS,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(COLS, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# Simulated RLHF training data
feedback_data = []

# Generate data for initial training
def generate_data(samples=5000):
    """
    Generates random board states for training.
    - X: Flattened board states.
    - y: Valid moves encoded as one-hot vectors.
    
    Why batches and epochs?
    - Batches split data into smaller chunks, allowing efficient training on large datasets.
    - Epochs define how many times the model will see the entire dataset during training.
    """
    X, y = [], []
    for _ in range(samples):
        board = np.random.choice([0, 1, -1], size=(ROWS, COLS), p=[0.8, 0.1, 0.1])
        X.append(board.flatten())
        valid_moves = [col for col in range(COLS) if board[0, col] == 0]
        move = np.random.choice(valid_moves)
        y.append(move)
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=COLS)
    return X, y

X_train, y_train = generate_data()
model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)

# RLHF logic: Save user feedback
def save_feedback(state, move, score):
    """
    Save user feedback for reinforcement learning (RLHF).
    - state: Current board state (flattened).
    - move: Column AI played.
    - score: User's evaluation (-1 = bad, 1 = good).
    
    Feedback helps the AI improve its decision-making using human preferences.
    """
    board = np.array(state).reshape(ROWS, COLS)
    feedback_data.append((board.flatten(), move, score))
    return f"Feedback saved! Move: {move}, Score: {score}"

# Update model with RLHF data
def retrain_model():
    """
    Retrain the model using user feedback.
    - Positive feedback: Reinforces good moves.
    - Negative feedback: Introduces randomness to discourage poor moves.
    """
    if not feedback_data:
        return "No feedback to train on!"
    X, y = [], []
    for board, move, score in feedback_data:
        X.append(board)
        y.append(move if score > 0 else np.random.choice(COLS))
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=COLS)
    model.fit(X, y, epochs=2, batch_size=32, verbose=1)
    feedback_data.clear()
    return "Model retrained with feedback!"

    
# Update model with RLHF data
def reset_model():
    """
    Reset the model.
    """
    model = create_model()
    X_train, y_train = generate_data()
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
    return "Model retrained with feedback!"


# AI prediction
def predict_move(board):
    """
    Predict AI's next move using the trained model.
    - Input: Flattened board.
    - Output: Most probable column and probabilities for all columns.
    """
    board_input = board.flatten().reshape(1, -1)
    probabilities = model.predict(board_input)
    move = np.argmax(probabilities)
    return move, probabilities

# Play move
def play_move(board, col, current_player):
    """
    Place a piece on the board for the given column and player.
    - current_player: 1 (human) or -1 (AI).
    """
    for row in range(ROWS - 1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = current_player
            return board, True
    return board, False

# Check win conditions
def check_win(board, player):
    """
    Check if the player has achieved a win (4 in a row in any direction).
    """
    for row in range(ROWS):
        for col in range(COLS):
            if (
                check_direction(board, row, col, 1, 0, player) or
                check_direction(board, row, col, 0, 1, player) or
                check_direction(board, row, col, 1, 1, player) or
                check_direction(board, row, col, 1, -1, player)
            ):
                return True
    return False

def check_direction(board, row, col, dr, dc, player):
    """
    Check a specific direction for a win.
    """
    count = 0
    for _ in range(4):
        if 0 <= row < ROWS and 0 <= col < COLS and board[row, col] == player:
            count += 1
        else:
            break
        row += dr
        col += dc
    return count >= 4

# Gradio interface
def game_interface(state, column, feedback=None):
    board = np.array(state).reshape(ROWS, COLS)
    current_player = 1  # Human
    result_msg = ""
    
    # Player's move
    board, valid = play_move(board, column, current_player)
    if not valid:
        return board.flatten().tolist(), "Invalid move! Try again.", None, None

    # Check win
    if check_win(board, current_player):
        return board.flatten().tolist(), "You win! 🎉", None, None

    # AI move
    ai_move, probabilities = predict_move(board)
    board, _ = play_move(board, ai_move, -1)

    # Check win for AI
    if check_win(board, -1):
        return board.flatten().tolist(), f"AI wins! 🤖 (AI predicted {probabilities})", ai_move, None

    # Save feedback if provided
    if feedback is not None:
        save_feedback(board, ai_move, feedback)

    return board.flatten().tolist(), f"AI played column {ai_move}. Your turn!", ai_move, probabilities.tolist()

# Reset game
def reset_game():
    return [0] * (ROWS * COLS), "Game reset. Your turn!"

# Board display function
def display_board(board):
    symbols = {0: "-", 1: "X", -1: "O"}
    return "\n".join([" ".join([symbols[cell] for cell in row]) for row in np.array(board).reshape(ROWS, COLS)])

# Gradio UI
def gradio_ui():
    board = [0] * (ROWS * COLS)
    with gr.Blocks() as app:
        gr.Markdown("# Vier Gewinnt AI with RLHF 🎮")
        state = gr.State(board)

        with gr.Tab("Game"):
            with gr.Row():
                board_display = gr.TextArea(value=display_board(board), interactive=False, label="Game Board")
            
            with gr.Row():
                column_slider = gr.Slider(0, COLS - 1, step=1, label="Your Move (Column)")
                move_button = gr.Button("Play Move")
                result_display = gr.Textbox(label="Result")
                ai_output = gr.TextArea(label="AI Probabilities", interactive=False) 

            with gr.Row():
                feedback_slider = gr.Slider(-1, 1, step=1, label="Feedback on AI's Move (-1: Bad, 1: Good)")
                feedback_button = gr.Button("Submit Feedback")
                feedback_result = gr.Textbox(label="Feedback Status")

            with gr.Row():
                retrain_button = gr.Button("Retrain Model with Feedback")
                retrain_result = gr.Textbox(label="Retrain Status")
                
                # Reset button
                reset_button = gr.Button("Reset Game")
                reset_model_button = gr.Button("Reset Model")
            
            move_button.click(
                game_interface,
                inputs=[state, column_slider],
                outputs=[state, result_display, feedback_result, ai_output]
            ).then(
                lambda s: display_board(s), inputs=state, outputs=board_display
            )
            
            feedback_button.click(
                save_feedback,
                inputs=[state, column_slider, feedback_slider],
                outputs=feedback_result
            )
            
            retrain_button.click(
                retrain_model,
                outputs=retrain_result
            )
             
            reset_model_button.click(
                reset_model,
                outputs=retrain_result
            )
            
            # Reset button logic
            reset_button.click(
                lambda: ([0] * (ROWS * COLS)),  # Reset state to empty board
                outputs=state
            ).then(
                lambda s: display_board(s), inputs=state, outputs=board_display
            ).then(
                lambda: "Game reset! Ready to play again.", outputs=result_display
            )

        with gr.Tab("Explanation"):
            gr.Markdown("""
---

# **Deep Dive into the Vier Gewinnt AI**

This explanation provides a detailed overview of how the Vier Gewinnt (Connect Four) AI works. It is designed to be beginner-friendly yet precise, offering enough background and technical details to understand the model and its functionality.

---

## **Why Build This Model?**

This program is an AI designed to play a simplified version of Vier Gewinnt (Connect Four). It predicts moves based on the current state of the board and continuously learns to improve its strategy using a technique called **Reinforcement Learning from Human Feedback (RLHF)**. The goal is to train an AI that makes intelligent decisions while playing against a human player.

The model is constructed using **TensorFlow's Sequential API**, which enables us to create a neural network by stacking layers one after the other. Each layer processes the data and passes it to the next layer, learning patterns in the data along the way.

---

## **Key Components of the Model**

### **1. Adam Optimizer**

The optimizer is the algorithm that adjusts the model's weights to minimize errors during training.

#### **Why Adam?**

- Adam (**Adaptive Moment Estimation**) is a widely used optimizer in machine learning. It combines the strengths of:
  - **AdaGrad**: Works well with sparse data.
  - **RMSProp**: Handles non-stationary objectives effectively.
- Adam adjusts the learning rate for each parameter dynamically, making it ideal for training.

#### **Advantages of Adam:**

- Automatically adapts the learning rate for faster convergence.
- Reduces the need for manual hyperparameter tuning.
- Works effectively in various scenarios, including games and language models.

---

### **2. Categorical Crossentropy Loss**

The **loss function** measures the difference between the AI's predicted moves and the correct moves. This helps the model learn better strategies over time.

#### **Why Categorical Crossentropy?**

- Connect Four has **7 possible actions** (one for each column), so this is a **categorical classification problem**.
- This loss function penalizes the AI when its predicted probabilities for the correct column are low.

#### **How It Works:**

If the correct move is column `3`, the AI assigns probabilities to all columns. The closer the probability for column `3` is to `1.0`, the lower the loss. The training process aims to **minimize this loss**.

---

### **3. Accuracy Metric**

The **accuracy** metric measures how often the AI predicts the correct move. It is calculated as:



Accuracy provides a straightforward way to track the AI's performance.

---

### **4. Dense Layers with ReLU Activation**

The neural network uses **Dense (fully connected) layers** to process the input data.

#### **Why Dense Layers?**

- Each neuron in a dense layer is connected to every neuron in the next layer, allowing the model to learn complex patterns.
- **64 neurons per layer** strike a balance between learning power and computational efficiency.

#### **Why ReLU Activation?**

The **ReLU (Rectified Linear Unit)** activation function introduces non-linearity, enabling the model to learn intricate patterns in the data. ReLU is defined as:



**Benefits of ReLU:**

- Prevents the problem of vanishing gradients (where the model stops learning).
- Efficient and simple to compute.

---

## **Training the Model**

### **Epochs and Batches**

- **Epochs:** One complete pass through the entire dataset. In this program, the model trains for **2 epochs**.
- **Batches:** The dataset is divided into smaller chunks (**batch size of 32**) to make training more efficient.

---

### **Reinforcement Learning from Human Feedback (RLHF)**

#### **What is RLHF?**

This is a method where humans guide the AI's learning by providing feedback. The AI improves its decision-making based on this feedback.

#### **How It Works:**

1. **Feedback:** After each move, humans provide feedback (-1 for bad, +1 for good).
2. **Data Storage:** This feedback, along with the board state and AI’s move, is saved.
3. **Retraining:** The AI retrains on this new data to improve its future performance.

---

## **Game Mechanics**

### **How the AI Predicts Moves**

1. The board is a 6x7 grid, flattened into a 1D array of 42 values.
   - Each cell is represented as:
     - `0` = empty,
     - `1` = human's piece,
     - `-1` = AI's piece.
2. The AI outputs **7 probabilities** (one for each column).
3. The AI selects the column with the **highest probability** as its move.

---

### **Win Detection**

The function `check_win` scans the board for 4 consecutive pieces in any direction:

- **Horizontal** (row-wise).
- **Vertical** (column-wise).
- **Diagonal** (\ or /).

The helper function `check_direction` moves in one direction and counts consecutive pieces for the current player.

---

## **Gradio Interface**

### **Interactive Features**

The Gradio interface allows users to:

1. Select a column for their move.
2. View the AI's probabilities and results.
3. Provide feedback (-1 to +1).
4. Retrain the AI with the feedback.
5. Reset the game anytime.

### **Reset Button Logic**

The reset button:

- Resets the board to an empty grid (all zeros).
- Updates the Gradio display to reflect the reset state.

            ---
            """)
    return app

gradio_ui().launch(debug=True)
 
