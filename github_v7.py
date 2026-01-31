import streamlit as st
import pandas as pd
import os
import uuid
import time
import matplotlib.pyplot as plt
from PIL import Image

# === 0. 路径适配 (适配 GitHub 部署环境) ===
# 获取当前脚本所在文件夹的绝对路径 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 题库图片存放在 experiment_pool 文件夹中 [cite: 7, 19]
POOL_DIR = os.path.join(CURRENT_DIR, 'experiment_pool') 
# 题库 CSV 必须放在仓库根目录下 [cite: 14, 21]
GOLD_CSV = os.path.join(CURRENT_DIR, 'boosted_final_detail.csv') 
# 结果保存路径 (v7 版本) [cite: 105, 109]
RESULT_CSV = os.path.join(CURRENT_DIR, 'doctor_study_results_v7.csv')

# 11 类疾病中文映射表 [cite: 56, 127]
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
        st.error(f"未找到题库文件: {GOLD_CSV}，请确认已上传至仓库根目录")
        return pd.DataFrame()
    df = pd.read_csv(GOLD_CSV)
    df['is_ai_right'] = df['真实病名'] == df['Top1_预测']
    # 强制平衡抽样：6个对，4个错 (陷阱题) [cite: 118, 119]
    right_pool = df[df['is_ai_right']].sample(6) 
    wrong_pool = df[~df['is_ai_right']].sample(4) 
    test_set = pd.concat([right_pool, wrong_pool]).sample(frac=1).reset_index(drop=True)
    test_set['true_cn'] = test_set['真实病名'].map(ID_TO_NAME)
    test_set['ai_cn'] = test_set['Top1_预测'].map(ID_TO_NAME)
    return test_set

if 'test_set' not in st.session_state:
    st.session_state.test_set = load_balanced_test_set()

# === 2. 模块一：医生画像 [cite: 75-80] ===
if st.session_state.step == 'profile':
    st.title("🩺 皮肤病 AI 辅助诊断实验 (v7)")
    if 'auto_id' not in st.session_state:
        st.session_state.auto_id = f"Dr_{uuid.uuid4().hex[:4].upper()}"
    
    with st.form("profile_form"):
        st.info(f"您的系统分配编号：**{st.session_state.auto_id}**")
        h_level = st.selectbox("医院等级", ["三甲医院", "二级医院", "社区医院/基层"]) # [cite: 77]
        y_work = st.selectbox("工作年限", ["<5年", "5-15年", ">15年"]) # [cite: 78]
        daily_p = st.selectbox("日均接诊量", ["<10人", "10-30人", ">30人"]) # [cite: 79]
        p_trust = st.slider("实验前对 AI 的信任度 (1-5 分)", 1, 5, 3) # [cite: 80]
        if st.form_submit_button("进入测试"):
            st.session_state.dr_info = {
                "doctor_id": st.session_state.auto_id, "hospital_level": h_level, 
                "work_years": y_work, "daily_patients": daily_p, "prior_ai_trust": p_trust
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
    if 'q_start' not in st.session_state: st.session_state.q_start = time.time() # [cite: 94]

    st.subheader(f"题目 {idx + 1} / 10")
    img_path = os.path.join(POOL_DIR, f"{row['image_id']}.jpg")

    col_img, col_ui = st.columns([1, 1])
    with col_img:
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.error(f"图片未找到: {row['image_id']}.jpg")

    with col_ui:
        # 第一阶段：独立诊断 [cite: 85-92]
        st.markdown("### 第一阶段：独立诊断")
        d1 = st.selectbox("首选 (Top-1)", ["请选择"] + CLASS_NAMES_CN, key=f"d1_{idx}")
        d2 = st.selectbox("次选 (Top-2)", ["请选择"] + CLASS_NAMES_CN, key=f"d2_{idx}")
        d3 = st.selectbox("备选 (Top-3)", ["请选择"] + CLASS_NAMES_CN, key=f"d3_{idx}")
        conf_i = st.slider("初始诊断信心 (1-10)", 1, 10, 5, key=f"ci_{idx}") # [cite: 106]

        ready = (d1 != "请选择" and d2 != "请选择" and d3 != "请选择")

        if not st.session_state.show_ai:
            if st.button("查看 AI 建议", disabled=not ready):
                st.session_state.time_base = time.time() - st.session_state.q_start # [cite: 94]
                st.session_state.temp_init = [d1, d2, d3, conf_i]
                st.session_state.show_ai = True
                st.rerun()
            if not ready: st.caption("请先完成三项诊断选择以解锁 AI")

        # 第二阶段：AI 介入决策 [cite: 97-104]
        if st.session_state.show_ai:
            st.markdown("---")
            st.info(f"💡 AI 的建议为：**{ai_cn}**")
            
            if ai_cn == d1:
                st.success("✨ 您的首选与 AI 预测一致！")
                st.session_state.action = "一致坚持"
            else:
                st.warning("🧐 AI 建议不同，请决策：")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("采纳 AI 建议"): st.session_state.action = "采纳AI建议"
                with b2:
                    if ai_cn not in st.session_state.temp_init[:3]:
                        if st.button("增加 AI 为第四选"): st.session_state.action = "增加AI意见"
                with b3:
                    if st.button("坚持独立见解"): st.session_state.action = "坚持原见"

            if 'action' in st.session_state:
                conf_f = st.slider("最终决策信心 (1-10)", 1, 10, 5, key=f"cf_{idx}") # [cite: 107]
                if st.button("确认提交并进入下一题"):
                    d_final_1 = ai_cn if st.session_state.action == "采纳AI建议" else d1
                    
                    # 全维度数据埋点 [cite: 105-111]
                    res = {
                        **st.session_state.dr_info, "image_id": row['image_id'], "true_label": true_cn, "ai_pred": ai_cn,
                        "d_init_1": d1, "is_init_t1_ok": (d1 == true_cn),
                        "is_init_t3_ok": (true_cn in st.session_state.temp_init[:3]),
                        "d_final_1": d_final_1, "is_final_t1_ok": (d_final_1 == true_cn),
                        "conf_gain": conf_f - conf_i, "action_taken": st.session_state.action,
                        "time_base": st.session_state.time_base,
                        "is_rescued": (d1 != true_cn and d_final_1 == true_cn),
                        "is_misled": (d1 == true_cn and d_final_1 != true_cn)
                    }
                    pd.DataFrame([res]).to_csv(RESULT_CSV, mode='a', header=not os.path.exists(RESULT_CSV), index=False)
                    st.session_state.user_results.append(res)
                    st.session_state.current_idx += 1
                    st.session_state.show_ai = False
                    del st.session_state.action
                    del st.session_state.q_start
                    st.rerun()

# === 4. 完成页：结果对账 [cite: 112-124] ===
elif st.session_state.step == 'feedback':
    st.title("🏁 测试完成！结果报告")
    rdf = pd.DataFrame(st.session_state.user_results)
    
    # 准确率对比柱状图 [cite: 124]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(['初始 T1', '最终 T1', '初始 T3', '最终 T3/4'], 
           [rdf['is_init_t1_ok'].mean(), rdf['is_final_t1_ok'].mean(), 
            rdf['is_init_t3_ok'].mean(), (rdf['is_final_t1_ok'] | rdf['is_init_t3_ok']).mean()], 
           color=['#3498db', '#2ecc71', '#e67e22', '#d35400'])
    ax.set_ylim(0, 1.1)
    st.pyplot(fig)

    st.write("### 诊断明细 (包含正确答案)")
    st.table(rdf[['image_id', 'true_label', 'ai_pred', 'd_final_1', 'is_final_t1_ok']])
    # 提供 CSV 下载，防止数据丢失
    csv = rdf.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载您的诊断结果记录", csv, "results.csv", "text/csv")