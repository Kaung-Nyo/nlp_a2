# Import required libraries
from dash import html
from dash import dcc
from dash import  html, Input, Output, State
import dash_bootstrap_components as dbc
from dash import Dash
from dash_chat import ChatComponent
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext, datasets, math
from tqdm import tqdm
from huggingface_hub import PyTorchModelHubMixin
import model
import safetensors

config = model.config

model = model.LSTMLanguageModel(**config)
model = model.from_pretrained("kaung-nyo-lwin/nlp_a2_lm")


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


# Create a dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                label="A2 - Language Model",
                children=[
                    html.H2(
                        children="Token Generation",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    dbc.Stack(
                        dcc.Textarea(
                            id="note",
                            value="""Note : This model is trained on Game of Throne books, please type the related content""",
                            style={
                                "width": "100%",
                                "height": 15,
                                "whiteSpace": "pre-line",
                                "textAlign": "center",
                            },
                            readOnly=False,
                        )
                    ),                    
                    html.Br(),
                   ChatComponent(
        id="chat-component",
        messages=[
            {"role": "assistant", "content": "Hello!, I am a fan of Game of Throne stories. You can ask me about that."},
        ],
    ),
                ],
            ),
        ]
    )
)

input_query, output_gen = [], []

dataset = datasets.load_dataset('kaung-nyo-lwin/game-of-throne-text')

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

#function to tokenize
tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  

#map the function to each example
tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer})

vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'], 
min_freq=3) 
vocab.insert_token('<unk>', 0)           
vocab.insert_token('<eos>', 1)            
vocab.set_default_index(vocab['<unk>'])

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

prompt = 'Iron throne is'
max_seq_len = 30
seed = 0
temperature = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


@app.callback(
    Output("chat-component", "messages"),
    Input("chat-component", "new_message"),
    State("chat-component", "messages"),
    prevent_initial_call=True,
)
def handle_chat(new_message, messages):
    if not new_message:
        return messages

    updated_messages = messages + [new_message]
    
    if new_message["role"] == "user":
        generation = generate(new_message["content"], max_seq_len, temperature, model, tokenizer, 
                          vocab, device, seed)
        bot_response = {"role": "assistant", "content": ' '.join(generation)}
        # bot_response = {"role": "assistant", "content": response.choices[0].message.content.strip()}
        return updated_messages + [bot_response]

    
    # updated_messages += new_message + " testing"

    return updated_messages

# Run the app
if __name__ == "__main__":
    app.run_server()
