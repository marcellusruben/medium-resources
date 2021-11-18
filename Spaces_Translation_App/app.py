import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk import tokenize
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("t5-base")

@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        
    return model

model = load_model()

st.sidebar.subheader('Select your source and target language below.')
source_lang = st.sidebar.selectbox("Source language",['English'])
target_lang = st.sidebar.selectbox("Target language",['German','French'])

st.title('Simple English ➡️ German/French Translation App')

st.write('This is a simple machine translation app that will translate\
         your English input text into German or French language\
         by leveraging a pre-trained [Text-To-Text Transfer Tranformers](https://arxiv.org/abs/1910.10683) model.')
         
st.write('You can see the source code to build this app in the \'Files and version\' tab.')

st.subheader('Input Text')
text = st.text_area(' ', height=200)

if text != '':
    
    prefix = 'translate '+str(source_lang)+' to '+str(target_lang)
    sentence_token =  tokenize.sent_tokenize(text)
    output = tokenizer([prefix+sentence for sentence in sentence_token], padding=True, return_tensors="pt")
    translated_id = model.generate(output["input_ids"], attention_mask=output['attention_mask'], max_length=100)
    translated_word = tokenizer.batch_decode(translated_id, skip_special_tokens=True)
    
    st.subheader('Translated Text')
    st.write(' '.join(translated_word))
