#NOTE: I BASICALLY JUST COPIED AND PASTED THE COLAB PARTS RELATED TO INDEXING

import pyterrier as pt
import pandas as pd
import json
import shutil
import os
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
from pyterrier_dr import FlexIndex, RetroMAE
import torch

print(torch.cuda.is_available())