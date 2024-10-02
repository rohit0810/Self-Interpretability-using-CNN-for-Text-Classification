import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train = pd.read_csv('/data2/home/pvishal/dlnlp_presentation/train1.csv', names=('class', 'text'))
test = pd.read_csv('/data2/home/pvishal/dlnlp_presentation/test1.csv', names=('class', 'text'))

# Drop first row if necessary
train.drop(0, inplace=True)
test.drop(0, inplace=True)

# Ensure 'class' column is of integer type
train['class'] = train['class'].astype(int)
test['class'] = test['class'].astype(int)

# Sample data from each class
df_class0 = train[train['class'] == 0]
df_class1 = train[train['class'] == 1]

sample_size = 3000  # Number of samples per class
sampled_class0 = df_class0.sample(n=sample_size, random_state=42)
sampled_class1 = df_class1.sample(n=sample_size, random_state=42)

# Combine and shuffle
sampled_dataset = pd.concat([sampled_class0, sampled_class1])
sampled_dataset = sampled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and labels
X_train = sampled_dataset['text']
y_train = sampled_dataset['class'].astype(int)
X_test = test['text']
y_test = test['class'].astype(int)

# Tokenization and sequence padding
max_vocab = 1000  # Maximum vocabulary size
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_seq_length = 300  # Maximum sequence length
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, padding='post', maxlen=max_seq_length)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, padding='post', maxlen=max_seq_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).to(device)
X_test_tensor = torch.tensor(X_test_padded, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model Definition
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, hidden_units):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: batch_size x seq_length
        x = self.embedding(x).permute(0, 2, 1)  # batch_size x embedding_dim x seq_length
        conv_out = torch.relu(self.conv(x))  # batch_size x num_filters x conv_output_length
        pooled_output = self.pool(conv_out).squeeze(dim=2)  # batch_size x num_filters
        x = torch.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).squeeze()
        return x, pooled_output, conv_out

# Define hyperparameters
vocab_size = max_vocab  # Vocabulary size
embedding_dim = 100  # Embedding dimension
input_length = max_seq_length  # Sequence length

num_filters_list = [3, 10, 15, 20, 30]  # Number of filters to try
hidden_units_list = [8, 16, 32, 50]  # Number of hidden units to try

accuracy_results = np.zeros((len(hidden_units_list), len(num_filters_list)))
auc_results = np.zeros((len(hidden_units_list), len(num_filters_list)))

# Training and Evaluation Loop
num_epochs = 10  # Number of epochs

for i, num_filters in enumerate(num_filters_list):
    for j, hidden_units in enumerate(hidden_units_list):
        # Initialize model, loss function, and optimizer
        model = TextCNN(vocab_size, embedding_dim, num_filters, hidden_units).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        model.train()
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the training data
        model.eval()
        with torch.no_grad():
            y_pred, pooled_output, conv_out = model(X_train_tensor)
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y_train_tensor.cpu().numpy()
            val_auc = roc_auc_score(y_true_np, y_pred_np)
            val_acc = ((y_pred_np > 0.5) == y_true_np).mean()

        # Store the results
        accuracy_results[j, i] = val_acc
        auc_results[j, i] = val_auc

        # Print the validation accuracy and AUC for this configuration
        print(f'Filters: {num_filters}, Hidden Units: {hidden_units}, Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}')

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot accuracy results
plt.subplot(1, 2, 1)
for idx, hidden_units in enumerate(hidden_units_list):
    plt.plot(num_filters_list, accuracy_results[idx], label=f'{hidden_units} Hidden Units', linestyle='--', marker='o')
plt.xlabel('Number of Filters')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Filters')
plt.legend()

# Plot AUC results
plt.subplot(1, 2, 2)
for idx, hidden_units in enumerate(hidden_units_list):
    plt.plot(num_filters_list, auc_results[idx], label=f'{hidden_units} Hidden Units', linestyle='--', marker='o')
plt.xlabel('Number of Filters')
plt.ylabel('Global AUC')
plt.title('Global AUC vs. Number of Filters')
plt.legend()

plt.tight_layout()
plt.show()

# Extract weights from the model
fc1_weights_array = model.fc1.weight.data.cpu().numpy()
fc1_biases_array = model.fc1.bias.data.cpu().numpy()
fc2_weights_array = model.fc2.weight.data.cpu().numpy()
fc2_biases_array = model.fc2.bias.data.cpu().numpy()
fc3_weights_array = model.fc3.weight.data.cpu().numpy()
fc3_biases_array = model.fc3.bias.data.cpu().numpy()

final_weights = [fc1_weights_array.T, fc2_weights_array.T, fc3_weights_array.T]
final_biases = [fc1_biases_array, fc2_biases_array, fc3_biases_array]

# Model Interpretation
from aletheia import UnwrapperClassifier

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# Convert data to NumPy arrays
X_train_numpy = tensor_to_numpy(pooled_output)
y_train_numpy = y_train_tensor.cpu().numpy().astype(int)

# Fit the UnwrapperClassifier
clf1 = UnwrapperClassifier(final_weights, final_biases)
clf1.fit(X_train_numpy, y_train_numpy)

# Summarize the model's local linear regions
clf1.summary()

# Generating Explanations
def generate_table(model, X_data, y_data, tokenizer, num_samples=5):
    model.eval()
    with torch.no_grad():
        predictions, pooled_outputs, conv_outs = model(X_data)
        important_filters = torch.topk(pooled_outputs, k=5, dim=1).indices.cpu().numpy()
        weights = model.fc1.weight.cpu().numpy()

        data = []
        for i in range(len(X_data)):
            sample_info = {
                "Sample ID": i,
                "Label": int(y_data[i].cpu().numpy()),
                "Predict": float(predictions[i].cpu().numpy()),
            }
            # Add top filters and phrases
            for j in range(5):  # Assuming we're using the top 5 filters
                filter_idx = important_filters[i, j]
                max_activation_idx = conv_outs[i, filter_idx].argmax().item()
                start_idx = max(0, max_activation_idx - 2)
                end_idx = start_idx + 5
                word_indices = X_data[i, start_idx:end_idx].cpu().numpy()
                phrase = " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in word_indices])
                sample_info[f"Filter {filter_idx} (β={weights[0, filter_idx]:.4f})"] = phrase

            data.append(sample_info)

        df = pd.DataFrame(data)
        # Sort columns so that the sample ID, label, and prediction come first
        df = df[["Sample ID", "Label", "Predict"] + [col for col in df.columns if col not in ["Sample ID", "Label", "Predict"]]]
        
        # Sort by Predict in ascending order and pick top samples
        df_ascending = df.sort_values(by="Predict", ascending=True).head(num_samples)
        
        # Sort by Predict in descending order and pick top samples
        df_descending = df.sort_values(by="Predict", ascending=False).head(num_samples)
        
    return df_ascending, df_descending

# Generate and display the tables
df_ascending, df_descending = generate_table(model, X_train_tensor, y_train_tensor, tokenizer, num_samples=5)

# Display the results in a neat format
pd.set_option('display.max_columns', None)  # To display all columns in the DataFrame
pd.set_option('display.width', 1000)  # To prevent wrapping in the output

print("Top 5 Samples with Lowest Predictions (Ascending):")
print(df_ascending)
print("\nTop 5 Samples with Highest Predictions (Descending):")
print(df_descending)
