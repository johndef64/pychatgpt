#%%
import os
import pychatgpt as op
import pyperclip as pc
print(os.getcwd())

#op.choose_model()

reply_type = {'latex': '''Reply only using Latex markup language. \nReply example:\n```latex\n\\documentclass{article}\n\n\\begin{document}\n\n\\section{basic LaTeX document structure}\nThis is a basic LaTeX document structure. In this example, we are creating a new document of the `article` class. The `\\begin{document}` and `\\end{document}` tags define the main content of the document, which can include text, equations, tables, figures, and more.\n\n\\end{document}\n```\n
''', 'code':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```python\n# Sure, I\'m here to help\n\ndef greeting(name):\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```\n'''}

delamain = "You are a coding copilot expert in any programming language."+reply_type['code']

watson = '''You are a scientific assistant, expert in paper publishing and scientific journals. Your main field are Biology and Informatics. '''+reply_type['latex']

robert = '''You are a scientific assistant, expert in python an R Bioinformatics. Your main fields are Biology and Informatics.'''+reply_type['code']

delamain = '''As a chatbot focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, PowerShell, R, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each language.\n4. Give less instructions as possible and only as comments in the code (# this is a comment).\n5. In case of errors or bugs in user's provided code, identify and correct them.\n6. provide explanations *only* if requested, provide just the requested code by default.\n7. writing code, be sure to comment it to give a clear understanding of what each section does.\n'''+reply_type['code']

roger = delamain+'''\nYou are a virtual assistant focused mainly on Bioinformatics, expert in R.'''
pyper = delamain+'''\nYou are a virtual assistant focused mainly on Python, expert in every python package'''

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