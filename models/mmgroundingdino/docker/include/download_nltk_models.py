import os
import shutil
import nltk

nltk_dir = '/cache/mmgroundingdino/nltk_data'

if not os.path.exists(nltk_dir):
    os.makedirs(nltk_dir)
if not os.listdir(nltk_dir):
    print('Downloading nltk models...')
    nltk.download('punkt', download_dir=nltk_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_dir)
else:
    print('nltk models already downloaded')
# copy contents of nltk_dir to '~/nltk_data'
target_dir = os.path.expanduser("~/nltk_data")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.copytree(nltk_dir, target_dir)
