import pandas as pd
# import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
def extract_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = " ".join([reader.getPage(i).extractText() for i in range(reader.numPages)])
    return text
def extract_words_from_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.astype(str)
    # Splitting each row's entry by spaces to get individual words
    return [entry.split() for entry in df.values.flatten()]
def extract_from_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()
def extract_data(file_path, file_format):
    if file_format == 'pdf':
        return extract_from_pdf(file_path)
    elif file_format == 'csv':
        return extract_words_from_csv(file_path)
    elif file_format == 'txt':
        return extract_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")
    
# def text_to_embedding(text):
#     input_ids = tokenizer.encode(text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(input_ids)
#     return outputs.last_hidden_state.mean(dim=1)

def text_to_embedding(text):
    # Set the padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the text with a higher max_length and generate the attention mask
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Use the embedding of the [CLS] token
    return outputs[0][:, 0, :]

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

def generate_remediation(log):
    input_text = f"Remediation for {log} compliance issue?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_relevant_policy_by_embedding(log_embedding, policy_data):
    # We assume that policy_data is a list of individual policies
    best_score = -float('inf')
    best_policy = ""

    for policy in policy_data:
        policy_embedding = text_to_embedding(policy)
        similarity_score = cosine_similarity(log_embedding, policy_embedding)

        if similarity_score > best_score:
            best_score = similarity_score
            best_policy = policy

    return best_policy



# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2)
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# Input
policy = "User must not attempt multiple failed logins."
log = "User 'jane_doe' attempted ten failed login."
question = "Is the log compliant with the policy?"

# Combine policy and log as text
text = f"Question: {question}\nPolicy: {policy}\nLog: {log}"

# Tokenize and classify
inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
outputs = model(**inputs)
logits = outputs.logits

# Apply softmax to get probabilities
probs = F.softmax(logits, dim=1)

# Get the predicted label and confidence score
predicted_label = torch.argmax(probs).item()
confidence_score = probs[0][predicted_label].item()

# Map predicted label to "Yes" or "No"
answer = "Yes" if predicted_label == 1 else "No"

print("Generated Answer:", answer)
print("Confidence Score:", confidence_score)


policy_data_list = extract_data('policies_questions.csv','csv')
logs_list = extract_data('logs.csv' ,'csv')
# Drop rows containing 'nan' values
policy_data_list = [policy_data for policy_data in policy_data_list if policy_data != ['nan']]
logs_list = [log_data for log_data in logs_list if log_data != ['nan']]
print(policy_data_list)
print(logs_list)

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2)

labels = labels_list
# Combine policy and log as text
policy_text = ' '.join(policy_list[0])  # Join the words to form the policy text

# Split data into train and validation sets
log_texts_train, log_texts_val, labels_train, labels_val = train_test_split(log_list[:], labels[:], test_size=0.2, random_state=42)

log_texts_train =log_texts_train + log_texts_val
labels_train = labels_train + labels_val


# Grid of hyperparameters
learning_rates = [1e-4]
batch_sizes = [40]

# Hyperparameter tuning loop
best_accuracy = 0.0
best_hyperparameters = {}

# Model Declaration
model = BertForSequenceClassification.from_pretrained(model_name, config=config)

# hyperparameters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rates = 1e-4
batch_sizes = 40
num_epochs = 20


# Create data loaders
train_inputs = tokenizer(log_texts_train, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
val_inputs = tokenizer(log_texts_val, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')

train_labels = torch.tensor(labels_train)
val_labels = torch.tensor(labels_val)

train_data = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
val_data = torch.utils.data.TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# Training loop

for epoch in tqdm(range(num_epochs)):
    # model = model.to('cuda')
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        print(loss)

    # Validation
    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            print(predicted,labels)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Learning Rate: {learning_rate} - Batch Size: {batch_size} - Accuracy: {accuracy:.4f}")



model.save_pretrained(best_model_dir)

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)
best_model_dir = "best_bert_model"  # Change this to the directory where you saved your best model
loaded_model = BertForSequenceClassification.from_pretrained(best_model_dir, config=config)

# Assuming policy_list and log_list contain your data

question = 'Is the log compliant with the policy?'
results = []  # To store the results
i = 0 
# Loop through policy and log pairs
for policy in newpolicy_data_list:
  for log in  newlogs_list:
    input_text = f"Policy: {policy}\nLog: {log}\nQuestion: {question}\n"
    inputs = tokenizer(input_text, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    i+=1
    print(i)

    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        
    results.append({"Policy": policy, "Log": log, "Predicted_Label": predicted_label})

# Create a DataFrame from the results
output_df = pd.DataFrame(results)

# Display the DataFrame
print(output_df)
