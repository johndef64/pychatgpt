#%%
import os
import pychatgpt as op
import pyperclip as pc
print(os.getcwd())

#op.choose_model()

#%% md
# Load file as variable
path = os.getcwd()
file = op.load_file(path)
file
#%% md
# # Load chat
#%%
if op.simple_bool('Load chat?'):
    op.load_chat()
    df = op.pd.DataFrame(op.chat_gpt)
    print('\n')
    for i in range(len(df)):
        print(df.role[i],':\n', df.content[i])
        print('----------')
else:
    op.chat_gpt = []
    print('*new chat*')
#%%
# expand chat
op.clearchat()
a= '' #'system'
b= '' #'assistant'
c= '' #'user'
if a != '': op.expand_chat(a, 'system')
if b != '': op.expand_chat(b, 'assistant')
if c != '': op.expand_chat(c, 'user')
op.chat_gpt
#%%
# continue chat
#op.clearchat()
system = '''  '''
message = ''' 

'''
op.send_message(message, system=system, model= op.model)
pc.copy(message+'\n'+op.reply)
pc.copy(op.reply)
#%%
pc.copy(op.reply)
#%%

#%%
m= '''

'''
op.send_message(m)
pc.copy(op.reply)
#%%
pc.copy(op.reply)
#op.save_chat()