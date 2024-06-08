import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned model and tokenizer
model_name_or_path = "./fine-tuned-haiku-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)

# Function to generate haiku
def generate_haiku(prompt, model, tokenizer, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(
    page_title="Haiku Generator",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🍃 Haiku Generator 🍃")
st.markdown("Welcome to the Haiku Generator! Enter the first line of your haiku and let our fine-tuned model create a poem for you.")

st.image("https://i.etsystatic.com/15034562/r/il/90d004/1644188658/il_794xN.1644188658_5tia.jpg", use_column_width=True)  # Add a relevant image (replace with actual image URL)

st.markdown(
    """
    **Instructions:**
    - Enter a single line as a prompt for the haiku.
    - Click on "Generate Haiku" to see the result.
    """
)

prompt = st.text_area("Enter the first line of your haiku:", placeholder="The white and scilent office...")

if st.button("Generate Haiku"):
    if prompt.strip():
        with st.spinner("Generating your haiku..."):
            haiku = generate_haiku(prompt, model, tokenizer)
        st.subheader("Your Generated Haiku:")
        st.write(f"\"{haiku}\"")
    else:
        st.warning("Please enter the first line of your haiku.")

st.markdown(
    """
    **About the Model:**
    This haiku generator is powered by a fine-tuned GPT-2 model, specifically trained on haiku poems to create coherent and artistic haikus based on your input.

    **Credits:**
    - Model fine-tuning: [Santiago Villasenor](https://github.com/SantiVillaRam7)
    - Model fine-tuning: [ElCachorroHumano](https://github.com/elcachorrohumano)
    - Model fine-tuning: [Oscar Martinez](https://github.com/Omarti34)
    - Haiku Dataset: [Haiku KTO](https://huggingface.co/datasets/davanstrien/haiku_kto)

    **Feedback:**
    If you have any suggestions or feedback, please feel free to [contact us](mailto:pablo.alazraki@itam.mx).
    """
)

st.sidebar.title("About")
st.sidebar.info(
    """
    This app generates haikus using a GPT-2 model fine-tuned on a haiku dataset.
    Explore the beauty of haiku poetry and get inspired by the creativity of AI.
    
    **GitHub Repository:**
    [FineTune-Haiku](https://github.com/SantiVillaRam7/FineTune-Haiku)
    
    **Powered by:**
    - [Streamlit](https://streamlit.io/)
    - [Hugging Face](https://huggingface.co/)
    """
)
