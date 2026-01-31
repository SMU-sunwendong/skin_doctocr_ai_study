import streamlit as st
import pandas as pd
import os
import uuid
import time
import matplotlib.pyplot as plt
from PIL import Image

# === 0. 路径与文件适配 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保文件名指向你最新的 v8 版本
GOLD_CSV = os.path.join(CURRENT_DIR, 'boosted_final_detail_v8.csv') 
POOL_DIR = os.path.join(CURRENT_DIR, 'experiment_pool')
RESULT_CSV = os.path.join(CURRENT_DIR, 'doctor_study_results_v7.csv')

# 11 类疾病中文映射表
ID_TO_NAME = {
    'MEL': '黑色素瘤', 'NV': '痣', 'BCC': '基底细胞癌', 'AK': '光化性角化病', 
    'BKL': '良性角化病', 'DF': '皮肤纤维瘤', 'VASC': '血管病变', 'SCC': '鳞状细胞癌', 
    'Vitiligo': '白癜风', 'Pityrasis-Alba': '白色糠疹', 'Psoriasis': '银屑病'
}
CLASS_NAMES_CN = list(ID_TO_NAME.values())

# === 1. 初始化状态机 ===
if 'step' not in st.session_state: st.session_state.step = 'profile'
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'show_ai' not in st.session_state: st.session_state.show_ai = False
if 'user_results' not in st.session_state: st.session_state.user_results = []

@st.cache_data
def load_balanced_test_set():
    if not os.path.exists(GOLD_CSV):
        st.error(f"未找到题库文件: {GOLD_CSV}")
        return pd.DataFrame()
    
    df = pd.read_csv(GOLD_CSV)
    # 核心逻辑：基于 v8 CSV 的 Top1 判定列
    df['is_ai_right'] = df['是否正确(Top1)'].str.upper() == 'YES'
    
    # 策略：随机抽取10道，若错题不足2道则强制补齐
    test_set = df.sample(n=min(len(df), 10))
    wrong_count = len(test_set[~test_set['is_ai_right']])
    
    if wrong_count < 2:
        wrong_pool = df[~df['is_ai_right']]
        right_pool = df[df['is_ai_right']]
        n_wrong = min(len(wrong_pool), 2)
        n_right = 10 - n_wrong
        test_set = pd.concat([right_pool.sample(n=n_right), wrong_pool.sample(n=n_wrong)])
    
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    test_set['true_cn'] = test_set['真实病名'].map(ID_TO_NAME)
    test_set['ai_cn'] = test_set['Top1_预测'].map(ID_TO_NAME)
    return test_set

if 'test_set' not in st.session_state:
    st.session_state.test_set = load_balanced_test_set()

# === 2. 模块一：医生画像 ===
if st.session_state.step == 'profile':
    st.title("🩺 皮肤病 AI 辅助诊断临床研究 (v7)")
    if 'auto_id' not in st.session_state:
        st.session_state.auto_id = f"Dr_{uuid.uuid4().hex[:4].upper()}"
    
    with st.form("profile_form"):
        st.info(f"系统分配编号：**{st.session_state.auto_id}
