#%%
import os
import requests
def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'wb') as file:
        file.write(response.content)

if not os.getcwd().endswith('pychatgpt.py'):
    handle="https://raw.githubusercontent.com/johndef64/pychatgpt/main/"
    files = ["pychatgpt.py","pychatgpt_static.py" ]
    for file in files:
        url = handle+file
        get_gitfile(url)

import pychatgpt as op
op.check_and_install_module('pyperclip')
import pandas as pd
import pyperclip as pc

#op.choose_model()

# Copilot Assistants:

features = {
    'reply_type' : {
        'latex': '''Reply only using Latex markup language. \nReply example:\n```latex\n\\documentclass{article}\n\n\\begin{document}\n\n\\section{basic LaTeX document structure}\nThis is a basic LaTeX document structure. In this example, we are creating a new document of the `article` class. The `\\begin{document}` and `\\end{document}` tags define the main content of the document, which can include text, equations, tables, figures, and more.\n\n\\end{document}\n```\n''',
        'python':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```python\n# Sure, I\'m here to help\n\ndef greeting(name):\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```\n''',
        'r':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```R\n# Sure, I\'m here to help\n\ngreeting <- function(name) {\n  # This function takes in a name as input and prints a greeting message\n  print(paste0("Hello, ", name, "!"))\n}\n\n# Prompt the user for their name\nuser_name <- readline(prompt = "What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```'''
    },

    'delamain' : '''As a chatbot focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, PowerShell, R, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each language.\n4. Give less instructions as possible and only as comments in the code (```# this is a comment```).\n5. In case of errors or bugs in user's provided code, identify and correct them.\n6. provide explanations *only* if requested, provide just the requested code by default.\n7. writing code, be sure to comment it to give a clear understanding of what each section does.\n'''
}

assistants = {
    'base': 'You are an helpful assistant.',
    'naive' : "You are a coding copilot expert in any programming language.\n"+features['reply_type']['python'],
    'watson' : '''You are a Scientific Assistant, expert in paper publishing and scientific journals (Elsevier, Springer, Nature, Science). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['latex'],
    'robert' : '''You are a scientific assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['r'],
    'delamain' : features['delamain'] + features['reply_type']['python'],
    'roger' : features['delamain']  + '''\nYou are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics.\n'''+features['reply_type']['r'],
    'pyper' : features['delamain']  + '''\nYou are a Virtual Assistant focused mainly on Python, expert in every python package'''+features['reply_type']['python']
}

def display_assistants():
    print('Available Assistants:')
    df = pd.DataFrame(assistants.items(), columns=['assistant', 'instructions'])
    display(df)

display_assistants()
#op.choose_model()
#%% Load file as variable
path = os.getcwd()
file = op.load_file(path)

#%% Load chat
op.load_chat()
df = op.pd.DataFrame(op.chat_gpt)
print(df.head())

#%% expand chat
op.clearchat()
#op.expand_chat('''  ''', 'system')
op.expand_chat('''  ''', 'assistant')
#op.expand_chat('''  ''', 'user')

#%% start/continue chat
#op.clearchat()
system = '''  '''
m = r"""

"""
m='''
don't make a column assistants but insted put that avlues as the index of the df
'''
op.send_message(m, system=assistants['delamain'], model= op.model)
pc.copy(m+'\n'+op.reply)
pc.copy(op.reply)
#%%
m = ''' 

'''
op.send_message(m, system=assistants['delamain'], model= op.model)
pc.copy(op.reply)
#%%

#%%
op.clearchat()
m = '''
What you can do?
'''
op.send_message(m, system=assistants['watson'], model= op.model)
pc.copy(op.reply)
#%%
m = '''

'''
op.send_message(m, system=assistants['watson'], model= op.model)
pc.copy(op.reply)
#op.save_chat()
#%%