# Streamlit_app.py
import streamlit as st
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
import pandas as pd
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM



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

st.markdown("""
<style>
body {
    background-image: url('https://www.transparenttextures.com/patterns/asfalt-dark.png');
}
.css-1n76uvr, .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>select, .avg-score, .stAlert>div {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #D0D0D0 !important;
}
.evaluate-box {
    background-color: rgba(0, 0, 0, 0.7);
    padding: 10px;
    border-radius: 15px;
}
.avg-score {
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
.title {
    font-family: 'Courier New', Courier, monospace;
    color: #FFD700;
    font-size: 2.5em;
    text-shadow: 3px 3px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
}
.eval-container {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>sudo h@ck</div>", unsafe_allow_html=True)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2)
model = BertForSequenceClassification.from_pretrained('best_bert_model', config=config)

st.title("Compliance Monitor")

# Start evaluation container with a transparent background
st.markdown("<div class='eval-container'>", unsafe_allow_html=True)

uploaded_policy_file = st.file_uploader("Upload the policies CSV file", type=["csv"])
uploaded_logs_file = st.file_uploader("Upload the logs CSV file", type=["csv"])

if uploaded_policy_file and uploaded_logs_file:
    policy_data_list = pd.read_csv(uploaded_policy_file).values.tolist()
    logs_list = pd.read_csv(uploaded_logs_file).values.tolist()

    remediation_data = pd.read_csv("/home/satyam/sudo_hack/grid/suggested_remediation.csv").values.tolist()
    remediation_map = {policy[0]: remediation[0] for policy, remediation in zip(policy_data_list, remediation_data)}

    policy_data_list = [policy_data for policy_data in policy_data_list if policy_data != ['nan']]
    logs_list = [log_data for log_data in logs_list if log_data != ['nan']]

    # with st.markdown("<div class='evaluate-box'>", unsafe_allow_html=True):
    if st.button("Evaluate"):
        with st.spinner('Evaluating...'):
            results = []
            total_logs = len(logs_list) * len(policy_data_list)
            progress_counter = 0
            progress_bar = st.progress(0)

            ip_pattern = r"\(IP: (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\)"
            for policy_data in policy_data_list:
                policy = ' '.join(policy_data)

                for log_data in logs_list:
                    log = ' '.join(log_data)
                    ip_match = re.search(ip_pattern, log)
                    ip_address = ip_match.group(1) if ip_match else "N/A"

                    text = f"Policy: {policy}\nLog: {log}"
                    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)
                    predicted_label = torch.argmax(probs).item()
                    confidence_score = probs[0][predicted_label].item()
                    answer = "Yes" if predicted_label == 1 else "No"

                    if predicted_label == 0:
                        results.append({"Policy": policy, "Log": log, "Answer": answer, "Confidence Score": confidence_score, "IP Address": ip_address, "Remediation": remediation_map[policy]})
                    
                    progress_counter += 1
                    progress_bar.progress(progress_counter / total_logs)

            st.success("Evaluation complete!")

            results_df = pd.DataFrame(results).sample(frac=1).reset_index(drop=True)
            def color_rows(val):
                color = 'red' if val == 'No' else 'green'
                return f'color: {color}'
            st.dataframe(results_df.style.applymap(color_rows, subset=['Answer']))

            avg_confidence_score = results_df["Confidence Score"].mean()
            st.markdown(f"<div class='avg-score'>Average Confidence Score: {avg_confidence_score:.2f}</div>", unsafe_allow_html=True)

            file_format = st.selectbox("Select file format for download", ["csv", "xlsx", "json"])
            if st.button("Download Results"):
                if file_format == "csv":
                    st.download_button("Download CSV", results_df.to_csv(index=False), mime="text/csv")
                elif file_format == "xlsx":
                    st.download_button("Download Excel", results_df.to_excel(index=False), mime="application/vnd.ms-excel")
                elif file_format == "json":
                    st.download_button("Download JSON", results_df.to_json(), mime="application/json")