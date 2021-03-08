import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *

predictors = pd.DataFrame({'development_age_in_months': 3, 'loss_ratio_segment': 'RENEWAL', 'ep': 232323}, index=[0])

model = load_model( './data/loss_ratio_seg_asof_20210228_acc_20210101')
value = predict_model(model, predictors)['Label'][0]

st.write(f'predicted variance is {value}')
