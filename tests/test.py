from pychatgpt import luca, michael, yoko, C, Cp, chatgpt, show_chat, pc, GPT

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
yoko('ciao Yoko come stai?')




#%% Alternative architecture:

# Crea un'istanza della classe GPT
op = GPT()
michael = GPT().michael
michael('@mi chiamo Giovanni ')
#%%
michael('Come mi chiamo Bro?')
#%%

#%%
# Usa il metodo galileo sull'istanza
op.julia('My name is John')
print()
op.julia('I live in New York City')
print()
op.julia('Who am I?')
#%%
op.chat_thread
#%%
print()
op.julia('What is this?', img=op.dummy_img)