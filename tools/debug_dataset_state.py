import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, os.getcwd())
from utils.session_manager import init_session, reset_on_new_dataset, get_dataframe, has_dataset, SessionKeys

init_session()

df = pd.read_csv('data/Iris.csv')
reset_on_new_dataset(df, 'Iris.csv')

print('Loaded Flag:', st.session_state.get(SessionKeys.DATASET_LOADED))
print('DF Exists:', 'df' in st.session_state and st.session_state.get('df') is not None)
print('UPLOADED_DF Exists:', SessionKeys.UPLOADED_DF in st.session_state and st.session_state.get(SessionKeys.UPLOADED_DF) is not None)
now = get_dataframe()
print('DF Shape:', now.shape if now is not None else None)
print('has_dataset():', has_dataset())
print('\n-- Raw keys snapshot --')
for k in [SessionKeys.UPLOADED_DF, SessionKeys.DF, 'df', SessionKeys.DATASET_LOADED]:
    print(k, ':', repr(st.session_state.get(k)))
