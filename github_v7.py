import streamlit as st
import pandas as pd
import os
import uuid
import time
import matplotlib.pyplot as plt
from PIL import Image

# === 0. 路径与文件适配 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 核心：使用你指定的 detail4 文件
GOLD_CSV = os.path.join(CURRENT_DIR, 'boosted_final_detail4.csv') 
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
    # 核心：根据 Top1_预测 和 真实病名 实时判定正确性
    df['is_ai_right'] = df['Top1_预测'] == df['真实病名']
    
    # 策略：随机抽取10道，若错题不足2道则强制补齐
    test_set = df.sample(n=min(len(df), 10))
    wrong_count = len(test_set[~test_set['is_ai_right']])
    
    if wrong_count < 2:
        wrong_pool = df[~df['is_ai_right']]
        right_pool = df[df['is_ai_right']]
        # 确保至少有 2 个错题，剩下的用对题补足 10 个
        n_wrong = min(len(wrong_pool), 2)
        n_right = 10 - n_wrong
        test_set = pd.concat([right_pool.sample(n=n_right), wrong_pool.sample(n=n_wrong)])
    
    # 打乱题目顺序
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
        st.info(f"系统分配编号：**{st.session_state.auto_id}**")
        h_level = st.selectbox("医院等级", ["三甲医院", "二级医院", "社区医院/基层"]) 
        y_work = st.selectbox("工作年限", ["<5年", "5-15年", ">15年"]) 
        p_trust = st.slider("实验前对 AI 的信任评分 (1-5 分)", 1, 5, 3) 
        if st.form_submit_button("进入测试"):
            st.session_state.dr_info = {
                "doctor_id": st.session_state.auto_id, "hospital_level": h_level, 
                "work_years": y_work, "prior_ai_trust": p_trust
            }
            st.session_state.step = 'test'
            st.rerun()

# === 3. 核心答题逻辑 ===
elif st.session_state.step == 'test':
    idx = st.session_state.current_idx
    if idx >= 10:
        st.session_state.step = 'feedback'
        st.rerun()

    row = st.session_state.test_set.iloc[idx]
    ai_cn, true_cn = row['ai_cn'], row['true_cn']
    if 'q_start' not in st.session_state: st.session_state.q_start = time.time()

    st.subheader(f"题目 {idx + 1} / 10")
    img_path = os.path.join(POOL_DIR, f"{row['image_id']}.jpg")

    col_img, col_ui = st.columns([1, 1])
    with col_img:
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.error(f"图片未找到: {row['image_id']}.jpg")

    with col_ui:
        st.markdown("### 第一阶段：独立诊断")
        d_init_1 = st.selectbox("首选 (Top-1)", ["请选择"] + CLASS_NAMES_CN, key=f"d1_{idx}")
        d_init_2 = st.selectbox("次选 (Top-2)", ["请选择"] + CLASS_NAMES_CN, key=f"d2_{idx}")
        d_init_3 = st.selectbox("备选 (Top-3)", ["请选择"] + CLASS_NAMES_CN, key=f"d3_{idx}")
        conf_init = st.slider("初始诊断信心 (1-10)", 1, 10, 5, key=f"ci_{idx}")

        ready = (d_init_1 != "请选择" and d_init_2 != "请选择" and d_init_3 != "请选择")

        if not st.session_state.show_ai:
            if st.button("查看 AI 建议", disabled=not ready):
                st.session_state.time_base = time.time() - st.session_state.q_start 
                st.session_state.temp_init = [d_init_1, d_init_2, d_init_3, conf_init]
                st.session_state.show_ai = True
                st.rerun()

        if st.session_state.show_ai:
            st.markdown("---")
            st.info(f"💡 AI 的建议为：**{ai_cn}**")
            
            if ai_cn == d_init_1:
                st.success("✨ 您的首选与 AI 预测一致！")
                st.session_state.action = "一致坚持"
            else:
                st.warning("🧐 AI 建议不同，请决策：")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("采纳 AI 建议"): st.session_state.action = "采纳AI建议"
                with b2:
                    if ai_cn not in st.session_state.temp_init[:3]:
                        if st.button("增加为第四选项"): st.session_state.action = "增加AI意见"
                with b3:
                    if st.button("坚持独立见解"): st.session_state.action = "坚持原见"

            if 'action' in st.session_state:
                conf_final = st.slider("最终决策信心 (1-10)", 1, 10, 5, key=f"cf_{idx}")
                if st.button("确认结果并进入下一题"):
                    d_final_1 = ai_cn if st.session_state.action == "采纳AI建议" else d_init_1
                    res = {
                        **st.session_state.dr_info, "image_id": row['image_id'], "true_label": true_cn, "ai_pred": ai_cn,
                        "d_init_1": d_init_1, "d_final_1": d_final_1, 
                        "is_final_t1_ok": (d_final_1 == true_cn), # 核心：判定最终Top1是否正确
                        "action": st.session_state.action, "conf_gain": conf_final - conf_init,
                        "time_base": st.session_state.time_base, 
                        "time_total": time.time() - st.session_state.q_start
                    }
                    pd.DataFrame([res]).to_csv(RESULT_CSV, mode='a', header=not os.path.exists(RESULT_CSV), index=False)
                    st.session_state.user_results.append(res)
                    st.session_state.current_idx += 1
                    st.session_state.show_ai = False
                    del st.session_state.action
                    del st.session_state.q_start
                    st.rerun()

# === 4. 完成页：对账表格 ===
elif st.session_state.step == 'feedback':
    st.title("🏁 测试完成！结果对账报告")
    rdf = pd.DataFrame(st.session_state.user_results)
    
    st.write("### 诊断明细 (核心对账数据)")
    # 展示项目：图片ID，真实答案，AI预测答案，最终答案，是否一致
    st.table(rdf[['image_id', 'true_label', 'ai_pred', 'd_final_1', 'is_final_t1_ok']])
    
    csv_data = rdf.to_csv(index=False).encode('utf-8')
    st.download_button("📥 点击下载实验数据 CSV", csv_data, "study_results.csv", "text/csv")
