from pychatgpt import julia, yoko, C, GPT


#%%
julia.model = 4
julia.chat('come stai oggi?')
#%%
julia.chat('Cosè questo?', img=julia.dummy_img)
#%%
julia.chat_thread
#%%
#%%
julia.create_image("Crea ua bella fan art di Sailor Mercury", model='dall-e-3')
#%%
yoko.model = 4
yoko.chat('@Che gusto di gelato ti piace?')

#%%

C.chatp('correggi questo codice: \n')
#%%

topo_gigio = GPT(assistant='Topo Gigio')
topo_gigio.chat('ciao come stai?')