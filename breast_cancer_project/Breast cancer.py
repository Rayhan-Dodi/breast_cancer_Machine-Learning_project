{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Deep Neural Network for Breast Cancer Classification (Modified)\n",
    "This notebook demonstrates building a deep neural network using PyTorch with some enhancements:\n",
    "- Two hidden layers\n",
    "- LeakyReLU activation\n",
    "- SGD optimizer with momentum\n",
    "- Early stopping\n",
    "- Accuracy and loss plots\n",
    "\n",
    "Dataset: Breast Cancer Wisconsin (Diagnostic)\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Import libraries\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n"
   ],
   "execution_count": 1,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Load and preprocess the data\n",
    "data = load_breast_cancer()\n",
    "X = data.data\n",
    "y = data.target\n",
    "\n",
    "# Train/test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Standardize features\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Convert to torch tensors\n",
    "X_train = torch.tensor(X_train, dtype=torch.float32)\n",
    "X_test = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_train = torch.tensor(y_train, dtype=torch.long)\n",
    "y_test = torch.tensor(y_test, dtype=torch.long)\n",
    "\n",
    "# Create datasets and dataloaders\n",
    "train_dataset = TensorDataset(X_train, y_train)\n",
    "test_dataset = TensorDataset(X_test, y_test)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)\n",
    "test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)\n"
   ],
   "execution_count": 2,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Define the neural network with 2 hidden layers and LeakyReLU\n",
    "class ClassificationNet(nn.Module):\n",
    "    def __init__(self, input_units=30, hidden_units1=64, hidden_units2=32, output_units=2):\n",
    "        super(ClassificationNet, self).__init__()\n",
    "        self.fc1 = nn.Linear(input_units, hidden_units1)\n",
    "        self.act1 = nn.LeakyReLU()\n",
    "        self.fc2 = nn.Linear(hidden_units1, hidden_units2)\n",
    "        self.act2 = nn.LeakyReLU()\n",
    "        self.fc3 = nn.Linear(hidden_units2, output_units)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.act1(self.fc1(x))\n",
    "        x = self.act2(self.fc2(x))\n",
    "        x = self.fc3(x)\n",
    "        return x\n",
    "\n",
    "# Instantiate model\n",
    "model = ClassificationNet()\n",
    "print(model)\n"
   ],
   "execution_count": 3,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ClassificationNet(\n",
      "  (fc1): Linear(in_features=30, out_features=64, bias=True)\n",
      "  (act1): LeakyReLU(negative_slope=0.01)\n",
      "  (fc2): Linear(in_features=64, out_features=32, bias=True)\n",
      "  (act2): LeakyReLU(negative_slope=0.01)\n",
      "  (fc3): Linear(in_features=32, out_features=2, bias=True)\n",
      ")\n"
     ]
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Define loss function and optimizer (SGD with momentum)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)\n"
   ],
   "execution_count": 4,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Training loop with early stopping\n",
    "num_epochs = 100\n",
    "patience = 10  # Early stopping patience\n",
    "best_loss = np.inf\n",
    "trigger_times = 0\n",
    "\n",
    "train_losses = []\n",
    "train_accuracies = []\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    correct = 0\n",
    "    total = 0\n",
    "\n",
    "    for inputs, labels in train_loader:\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        running_loss += loss.item() * inputs.size(0)\n",
    "        _, predicted = torch.max(outputs.data, 1)\n",
    "        total += labels.size(0)\n",
    "        correct += (predicted == labels).sum().item()\n",
    "\n",
    "    epoch_loss = running_loss / total\n",
    "    epoch_acc = correct / total\n",
    "    train_losses.append(epoch_loss)\n",
    "    train_accuracies.append(epoch_acc)\n",
    "\n",
    "    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')\n",
    "\n",
    "    # Early stopping check\n",
    "    if epoch_loss < best_loss:\n",
    "        best_loss = epoch_loss\n",
    "        trigger_times = 0\n",
    "    else:\n",
    "        trigger_times += 1\n",
    "        if trigger_times >= patience:\n",
    "            print(f'Early stopping at epoch {epoch+1}')\n",
    "            break\n"
   ],
   "execution_count": 5,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100, Loss: 0.6749, Accuracy: 0.6143\n",
      "Epoch 2/100, Loss: 0.6132, Accuracy: 0.7131\n",
      "Epoch 3/100, Loss: 0.5732, Accuracy: 0.7546\n",
      "Epoch 4/100, Loss: 0.5406, Accuracy: 0.7874\n",
      "Epoch 5/100, Loss: 0.5097, Accuracy: 0.8227\n",
      "Epoch 6/100, Loss: 0.4794, Accuracy: 0.8523\n",
      "Epoch 7/100, Loss: 0.4465, Accuracy: 0.8779\n",
      "Epoch 8/100, Loss: 0.4174, Accuracy: 0.8996\n",
      "Epoch 9/100, Loss: 0.3938, Accuracy: 0.9170\n",
      "Epoch 10/100, Loss: 0.3686, Accuracy: 0.9318\n",
      "... (truncated for brevity) ...\n"
     ]
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Evaluate on test set\n",
    "model.eval()\n",
    "correct = 0\n",
    "total = 0\n",
    "with torch.no_grad():\n",
    "    for inputs, labels in test_loader:\n",
    "        outputs = model(inputs)\n",
    "        _, predicted = torch.max(outputs, 1)\n",
    "        total += labels.size(0)\n",
    "        correct += (predicted == labels).sum().item()\n",
    "\n",
    "print(f'Test Accuracy: {correct / total:.4f}')"
   ],
   "execution_count": 6,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Accuracy: 0.9474\n"
     ]
    }
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Plot training loss and accuracy\n",
    "plt.figure(figsize=(12,5))\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(train_losses, label='Training Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Training Loss Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "plt.plot(train_accuracies, label='Training Accuracy', color='orange')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Training Accuracy Over Epochs')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ],
   "execution_count": 7,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exercises\n",
    "1. Try increasing the number of neurons in the hidden layers.\n",
    "2. Experiment with other activation functions like `nn.ReLU` or `nn.Sigmoid`.\n",
    "3. Try training on the Iris dataset following the same pattern.\n",
    "4. Implement a validation split and monitor validation loss for early stopping.\n",
    "5. Try adding dropout layers to reduce overfitting."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
