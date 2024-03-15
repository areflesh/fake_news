
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from transformers import DistilBertForSequenceClassification, AdamW
import torch
def train_model_random_forest(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def train_model_mlp(X_train, y_train):

    model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)

    return model

def train_model_transformer(X_train_ids, X_train_masks, y_train_tensors, epochs, batch_size, peft_factor=0.1):
    """Train a parameter-efficient transformer model."""
    # Initialize the DistilBERT model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,  # Assuming binary classification (fake vs real)
        output_attentions=False,
        output_hidden_states=False,
    )

    # Reduce the number of trainable parameters by the specified factor
    for param in model.parameters():
        param.requires_grad = False

    # Calculate the number of trainable parameters to keep
    num_total_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = int(num_total_params * peft_factor)

    # Make the specified number of parameters trainable
    for param in model.parameters():
        if num_trainable_params > 0:
            param.requires_grad = True
            num_trainable_params -= param.numel()
        else:
            break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    total_steps = len(X_train_ids) // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step in range(0, len(X_train_ids), batch_size):
            batch_ids = X_train_ids[step:step+batch_size].to(device)
            batch_masks = X_train_masks[step:step+batch_size].to(device)
            batch_labels = y_train_tensors[step:step+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=batch_ids, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss:.4f}')
        avg_train_loss = total_loss / total_steps
        print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}')
    
    torch.save(model.state_dict(), './models/transformer')

    return model

