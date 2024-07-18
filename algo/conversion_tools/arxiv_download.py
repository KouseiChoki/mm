'''
Author: Qing Hong
Date: 2023-07-14 14:41:24
LastEditors: QingHong
LastEditTime: 2023-07-14 14:48:21
Description: file content
'''
import urllib.request

# The URL of the source files.
url = 'https://arxiv.org/pdf/2210.16900.pdf'

# The local path where the source files will be saved.
path = '/Users/qhong/Downloads/0711/1.tar.gz'

# Download the file from `url` and save it locally under `path`:
urllib.request.urlretrieve(url, path)



import re

# Open the LaTeX file
with open('/Users/qhong/Downloads/2210.16900/report.tex', 'r') as f:
    content = f.read()

# Remove LaTeX commands
content_no_commands = re.sub(r'\\.*?\\', '', content)

# Remove comments
content_no_comments = re.sub(r'%.*?\n', '', content_no_commands)

# Print the resulting text
print(content_no_comments)