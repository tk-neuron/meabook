#!/usr/bin/env python
# coding: utf-8

# # Datasets

# ## Download

# 本ノートブックで使用するデータはGithubからダウンロードできるが，以下にPythonにより自動でダウンロードする方法を示す（推奨）．

# In[1]:


from pathlib import Path
import urllib.request


# 以下はダウンロードするファイルの一覧である．

# In[2]:


baseurl = 'https://tk-neuron.github.io/meabook'

# sub-directory: files
files = {
    '01': ['mapping.csv', 'spikes.csv'],
    '02': ['spikes_unit.csv'],
    '03': [],
    '04': ['data.raw.h5'],
}


# デフォルトでは，このコードを実行する親ディレクトリ直下に`datasets/`ディレクトリを作り，その配下にサブディレクトリ `01`~`04`を作り，ファイル一覧をダウンロードする設定になっている．`datasets/`ディレクトリのパスは適宜変更可能．

# In[3]:


# create dataset directory (if already exists, pass)
datadir = Path.joinpath(Path.cwd().parent, 'datasets')
print('root for dataset: ', datadir, '\n')
Path(datadir).mkdir(exist_ok=True)

# create sub-directories and download each file
for key, values in files.items():
    subdir = Path.joinpath(datadir, key)
    print('sub-directory: ', subdir)
    Path(subdir).mkdir(exist_ok=True)
    
    for value in values:
        url = '/'.join([baseurl, key, value])
        filename = '/'.join([str(datadir), key, value])
        
        print(f'downloading file: {value} from {url}.')
        print('file saved in: ', filename)
        urllib.request.urlretrieve(url, filename)
        
    print('\n')


# ## For Colab Users

# Python実行環境としてColabを用いる場合は，データセットをGoogle Driveを通じてやりとりする必要がある．この場合，マウントという特殊な手順が必要になる．
# 
# まず，右上のロケットマークからOpen in Colabをクリックし，Colabのページを開く．
# 
# ```{admonition} Mount手順
# 1. 左のメニューバーからFileアイコンをクリックし，Driveのアイコン付きのボタン（Mount Drive）をクリック．
#     または次のコードセルを実行する．
#     ```python
#     from google.colab import drive
#     drive.mount('/content/drive')
#     ```
# 
# 2. "This notebook was not authored by Google" という警告が出るので， "Run Anyway" を選択し， "Connect to Google Drive" へ．
# 
# 3. すると，別ウィンドウでGoogleの認証画面が出るので，アカウントを選択してログイン． "Google Drive for desktop wants to access your Google Account" の画面で "Allow" をクリック．
# 
# 4. "Mounted at /content/drive" が表示されたら完了．
# ```

# マウントが完了すると，Google Drive上のファイルパスをローカルと同じように指定することができる．例えばデータセットをダウンロードするディレクトリを以下のようにMy Drive直下に設定することが可能．

# In[4]:


datadir = Path.joinpath(Path('/content/drive/MyDrive/'), 'datasets')
datadir

