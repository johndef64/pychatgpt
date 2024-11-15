from pychatgpt import luca, michael, yoko, C, Cp, chatgpt, show_chat, pc
#%%
def test_assistant(assistant_, m):
    if assistant_ == 'luca':
        luca(m)
    elif assistant_ == 'michael':
        michael(m)
    elif assistant_ == 'yoko':
        yoko(m)
    elif assistant_ == 'copilot':
        C(m)
    else:
        chatgpt(m)
#%%

test_assistant('michael', 'ciao Michael, come stai oggi?')
#%%
show_chat()
#%%


#%% Alternative architecture:
from pychatgpt.main_class import GPT
# Crea un'istanza della classe GPT
op = GPT()
#%%
# Usa il metodo galileo sull'istanza
op.julia('My name is John')
print()
op.julia('I live in New York City')
print()
op.julia('Who am I?')
#%%
print()
op.julia('What is this?', img=op.dummy_img)