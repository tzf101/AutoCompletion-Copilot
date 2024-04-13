import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chat Application with GEMMA-2B")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input widget
prompt = st.chat_input("Type your message here...")
access_token = "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"

# Load the tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=access_token)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    config=quantization_config,
    use_auth_token='your_access_token_here',
    device_map="auto"
)

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response using the model
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids['input_ids'], max_length=200, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
