from pychatgpt import julia, yoko, watson, C, GPT

#%%
julia.model = 4
julia.chat('how are you today?')
#%%
julia.chat("What's on the agenda today?")
#%%
julia.chat('What is this?', img=julia.dummy_img)
#%%
julia.chat_thread
#%%
#%%
julia.create_image("Create a Sailor Mercury fan art in punk goth style", model='dall-e-3')
#%%
yoko.model = 4
yoko.chat('@What flavour of ice cream do you like?')

#%%
watson.model = 4
watson.chat('write an abstract of a made-up scientific paper')
#%%
watson.format = 'markdown'
watson.chat('write an abstract of a made-up scientific paper')
#%%
# Copy your code into Clipboard
C.chatp('correct this code: \n')
#%%
C.chatp('@complete this code: \n')
#%%
C.chatp("@ save an empy json  with an empy list [] inside:\n\n")
#%%
# Custom Assistants
sailor_mercury = GPT(assistant='Sailor Mercury')
sailor_mercury.chat('Hi! How are you today?')

