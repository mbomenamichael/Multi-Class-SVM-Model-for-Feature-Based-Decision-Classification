# Multi-Class SVM Model for Feature-Based Decision Classification

A machine learning project implementing a **multi-class Support Vector Machine (SVM)** classifier from scratch to predict optimal actions in a game environment using feature vectors and supervised learning.

## About

This project implements a **multi-class Support Vector Machine (SVM) classifier** to predict optimal actions in a game environment.

The model learns from training data representing game states and corresponding actions. Using these examples, the classifier learns a decision boundary that separates different action classes and predicts the best move for new unseen states.

The SVM model is implemented **from scratch using Python and NumPy**, including the hinge loss function and gradient descent optimisation.

## Machine Learning Approach

The project uses a **multi-class Support Vector Machine (SVM)** model.

SVMs work by finding an optimal hyperplane that separates different classes in feature space while maximising the margin between them.

The classifier:

1. Converts game states into feature vectors
2. Learns weight parameters during training
3. Calculates class scores for new inputs
4. Predicts the class with the highest score

## Loss Function

The model uses **hinge loss**, a standard loss function used in Support Vector Machines.

Hinge loss penalises predictions where the correct class score is not sufficiently greater than incorrect class scores.

Loss is calculated as:

max(0, score_wrong − score_correct + margin)

This encourages the model to push correct class scores higher than incorrect class scores by a defined margin.

## Training Process

The classifier is trained using **gradient descent optimisation**.

Training steps include:

1. Initialise a weight matrix randomly
2. Calculate classification scores
3. Compute hinge loss
4. Calculate gradients
5. Update weights to reduce loss

The process repeats for multiple iterations until the model converges to an optimal solution.

## Dataset

The training dataset contains examples of game states and corresponding actions.

Each training example includes:

- Feature vector representing the game state
- Target label representing the correct move

Possible move classes include:

- North
- East
- South
- West

## Feature Representation

Game states are converted into **feature vectors** representing the environment.

Features describe elements of the game state such as:

- Pacman position
- Ghost proximity
- Food location
- Environmental state

These feature vectors are used as input to the SVM model.

## Project Structure

project/
│
├── classifierAgents.py       # Base classifier agent framework
├── classifierAgentsSVM.py    # SVM implementation and training logic
├── pacman environment files  # Game environment and utilities
├── good-moves.txt            # Training dataset
└── README.md

## Technologies Used

- Python
- NumPy
- Machine Learning
- Support Vector Machines
- Gradient Descent Optimisation
- Reinforcement Learning Environment

## Example Workflow

Training and prediction pipeline:

Game State
      ↓
Feature Extraction
      ↓
SVM Model
      ↓
Class Score Calculation
      ↓
Action Prediction
      ↓
Execute Move in Environment

## Learning Outcomes

This project demonstrates:

- Implementation of Support Vector Machines from scratch
- Hinge loss optimisation
- Gradient descent training
- Multi-class classification
- Machine learning for decision-making systems
