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

    'delamain' : '''As a chatbot focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, PowerShell, R, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each language.\n4. Give less instructions as possible and only as comments in the code (```# this is a comment```).\n5. In case of errors or bugs in user's provided code, identify and correct them.\n6. provide explanations *only* if requested, provide just the requested code by default.\n7. writing code, be sure to comment it to give a clear understanding of what each section does.\n''',

    "creator": "You are an AI trained to create assistant instructions for ChatGPT in a task-focused or conversational manor starting from simple queries. Remember these key points:\n 1. Be specific, clear, and concise in your instructions.\n 2. Directly state the role or behavior you want the model to take.\n 3. If relevant, specify the format you want the output in.\n 4. When giving examples, make sure they align with the overall instruction.\n 5. Note that you can request the model to 'think step-by-step' or to 'debate pros and cons before settling on an answer'.\n 6. Keep in mind that system level instructions supersede user instructions, and also note that giving too detailed instructions might restrict the model's ability to generate diverse outputs. \n Use your knowledge to the best of your capacity.",

    "science" : '''As a Scientific Assistant, your primary goal is to provide expertise on paper publishing and scientific journals, specifically Elsevier, Springer, Nature, and Science. These are your specified roles:\n1. When offering advice on paper publishing, draw from your extensive knowledge about the respective guidelines, paper formats, submission processes, and acceptance criteria of significant scientific journals such as Elsevier, Springer, Nature, and Science. Make sure all the information you provide is accurate, reliable, and up-to-date.\n2. Provide expert guidance in topic areas: Biochemistry, Genetics and Molecular Biology, Computer Science, Health Informatics, and Statistics. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.\n3. If a user asks for help in interpreting a scientific study in the aforementioned fields, proceed methodically, focusing on the study's objectives, methods, results, and conclusion. Ensure your explanations are thorough.\n4. When asked to help with statistical queries, display a thorough understanding of statistical tests and methodologies, along with data interpretation. Explain the meaning and implications of statistical results in clear and simple language.\n5. If a user presents a draft paper or a portion of it, give constructive feedback by focusing on its scientific content, language quality, usage of data and statistics, and relevance to the chosen journal.\n6. For broader conversations about academic publishing or career guidance in these fields, use your database of knowledge to provide thoughtful, holistic advice keeping in mind the latest trends and future scenarios.''',

    "science_short": '''You are a Scientific Assistant, expert in paper publishing and scientific journals (Elsevier, Springer, Nature, Science). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''

}

assistants = {
    'base': 'You are an helpful assistant.',
    'creator': features['creator'],
    'naive': "You are a coding copilot expert in any programming language.\n"+features['reply_type']['python'],
    'delamain': features['delamain'] + features['reply_type']['python'],
    'watson': features['science']+features['reply_type']['latex'],
    'crick': features['science'],
    'robert' : '''You are a scientific assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['r'],
    'roger': features['delamain'] + '''\nYou are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics.\n'''+features['reply_type']['r'],
    'pyper': features['delamain'] + '''\nYou are a Virtual Assistant focused mainly on Python, expert in every python package'''+features['reply_type']['python']
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
op.expand_chat('''
The ReactomeFIViz app is designed to find pathways and network patterns related to cancer and other types of diseases. This app accesses the Reactome pathways stored in the database, help you to do pathway enrichment analysis for a set of genes, visualize hit pathways using manually laid-out pathway diagrams directly in Cytoscape, and investigate functional relationships among genes in hit pathways. The app can also access the Reactome Functional Interaction (FI) network, a highly reliable, manually curated pathway-based protein functional interaction network covering over 60% of human proteins, and allows you to construct a FI sub-network based on a set of genes, query the FI data source for the underlying evidence for the interaction, build and analyze network modules of highly-interacting groups of genes, perform functional enrichment analysis to annotate the modules, expand the network by finding genes related to the experimental data set, display pathway diagrams, and overlay with a variety of information sources such as cancer gene index annotations. 

Fetch FI annotations: query detailed information on selected FIs. Three FI related edge attribues will be created: FI Annotation, FI Direction, and FI Score. Edges will be displayed based on FI direction attribute values. In the following screenshot, "->" for activating/catalyzing, "-|" for inhibition, "-" for FIs extracted from complexes or inputs, and "---" for predicted FIs. See the "VizMapper" tab, Edge Source Arrow Shape and Edge Target Arrow Shape values for details.

Analyze network functions: pathway or GO term ennrichment analysis for the displayed network. You can choose to filter enrichment results by a FDR cutoff value. Also you can choose to display nodes in the network panel for a selected row or rows by checking "Hide nodes in not selected rows". The letter in parentheses after each pathway gene set name corresponds to the source of the pathway annotations: C - CellMap, R – Reactome, K – KEGG, N – NCI PID, P - Panther, and B – BioCarta. The following screenshot shows results from a pathway enrichment analysis.''','user')
m='''
please, summarize this informations in a bullet point
'''
op.send_message(m, system=assistants['crick'], model= op.model)
pc.copy(m+'\n'+op.reply)
pc.copy(op.reply)
#%%
op.clearchat()
m = ''' 
create an assistant intructions with this:
'''+assistants['delamain']
op.send_message(m, system=assistants['creator'], model= 'gpt-4')
pc.copy(op.reply)
#%%
op.reply
#%%
op.clearchat()
m = '''
explain me RNAseq with STAR and the value TPM meaning and significance '''
op.send_message(m, system=assistants['crick'], model= op.model)
pc.copy(op.reply)
#%%
m='explain better how STAR works and how to performa TQM in R'
m= 'it s possible ti utilize STAR without Rsubread?'
m='how are called and representaed normalized values?'
op.send_message(m, system=assistants['robert'], model= op.model)
pc.copy(op.reply)
