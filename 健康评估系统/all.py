import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------- 作品一：健康系统评估 ---------
def health_system_app():
    st.title("作品一（健康系统评估）")
    st.markdown("请填写以下信息进行基本健康评估。")

    with st.form("health_form"):
        st.header("个人信息")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("姓名", value="张三")
            age = st.number_input("年龄", min_value=1, max_value=120, value=25)
            gender = st.selectbox("性别", ["男", "女"])
        with col2:
            height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=170.0)
            weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=65.0)

        st.header("健康指标")
        col3, col4 = st.columns(2)
        with col3:
            sbp = st.number_input("收缩压 SBP (mmHg)", min_value=50, max_value=250, value=120)
            hr = st.number_input("静息心率 (bpm)", min_value=30, max_value=200, value=70)
        with col4:
            dbp = st.number_input("舒张压 DBP (mmHg)", min_value=30, max_value=150, value=80)
            waist = st.number_input("腰围 (cm)", min_value=30.0, max_value=200.0, value=75.0)

        submit = st.form_submit_button("开始评估")

    if submit:
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        if bmi < 18.5:
            bmi_status, bmi_color = "偏瘦", "blue"
        elif 18.5 <= bmi < 24:
            bmi_status, bmi_color = "正常", "green"
        elif 24 <= bmi < 28:
            bmi_status, bmi_color = "超重", "orange"
        else:
            bmi_status, bmi_color = "肥胖", "red"

        if sbp >= 140 or dbp >= 90:
            bp_status, bp_color = "高血压", "red"
        elif sbp < 90 or dbp < 60:
            bp_status, bp_color = "低血压", "orange"
        else:
            bp_status, bp_color = "正常", "green"

        hr_status = "偏慢" if hr < 60 else "偏快" if hr > 100 else "正常"
        waist_flag = "腹型肥胖" if (gender == "男" and waist >= 90) or (gender == "女" and waist >= 85) else "正常"

        st.header("健康评估报告")
        st.markdown(f"**姓名**：{name}  \n**性别**：{gender}  \n**年龄**：{age} 岁")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown(f"**BMI：{bmi:.2f}**")
            st.markdown(f"体重分类：<span style='color:{bmi_color}'>{bmi_status}</span>", unsafe_allow_html=True)
        with col6:
            st.markdown(f"**血压：{sbp}/{dbp} mmHg**")
            st.markdown(f"血压分类：<span style='color:{bp_color}'>{bp_status}</span>", unsafe_allow_html=True)

        st.markdown(f"**心率：{hr} bpm（{hr_status}）**")
        st.markdown(f"**腰围：{waist} cm（{waist_flag}）**")

        if bmi_status == "正常" and bp_status == "正常" and hr_status == "正常" and waist_flag == "正常":
            st.success("身体状况良好，请继续保持良好的生活习惯。")
        else:
            st.info("建议保持合理饮食、控制血压、规律运动，并定期体检。")

        # BMI 分布图
        fig, ax = plt.subplots(figsize=(8, 1.8))
        ax.set_xlim(10, 40)
        ax.set_ylim(0, 1)
        bmi_ranges = [(10, 18.5, 'lightblue', '偏瘦'), (18.5, 24, 'lightgreen', '正常'),
                      (24, 28, 'orange', '超重'), (28, 40, 'red', '肥胖')]
        for start, end, color, label in bmi_ranges:
            ax.axvspan(start, end, color=color, alpha=0.4)
            ax.text((start + end) / 2, 0.5, label, ha='center', va='center', fontsize=10)
        ax.axvline(bmi, color='black', linewidth=2)
        ax.text(bmi, 0.9, f"BMI={bmi:.1f}", rotation=90, va='bottom', ha='center', fontsize=9, color='black')
        ax.set_yticks([])
        ax.set_xlabel("BMI 值")
        ax.set_title("BMI 区间分布图")
        st.pyplot(fig)

# --------- 作品二：糖尿病风险评估 ---------
@st.cache_resource
def train_diabetes_model():
    diabetes = load_diabetes()
    X = diabetes.data
    y = (diabetes.target > diabetes.target.mean()).astype(int)
    feature_names = diabetes.feature_names
    feature_labels = {
        'age': '年龄（标准化）',
        'sex': '性别（标准化）',
        'bmi': '身体质量指数 BMI（标准化）',
        'bp': '平均血压（标准化）',
        's1': '总胆固醇（标准化）',
        's2': '低密度脂蛋白 LDL（标准化）',
        's3': '高密度脂蛋白 HDL（标准化）',
        's4': '甘油三酯（标准化）',
        's5': '血清胰岛素（标准化）',
        's6': '血清胰岛素变异（标准化）'
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_labels': feature_labels
    }

def diabetes_model_app():
    st.title("作品二（糖尿病风险评估）")
    st.write("请输入您的身体指标：")

    bundle = train_diabetes_model()
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    feature_labels = bundle['feature_labels']

    user_input = []
    for fname in feature_names:
        label = feature_labels.get(fname, fname)
        val = st.text_input(f"{label}（数字）", "")
        user_input.append(val)

    if st.button("评估风险"):
        try:
            input_data = np.array([float(x) for x in user_input]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            pred_prob = model.predict_proba(input_scaled)[0][1]
            pred_label = model.predict(input_scaled)[0]
            st.write(f"预测结果：{'高风险（可能患糖尿病）' if pred_label == 1 else '低风险（患病可能性较低）'}")
            st.write(f"患病概率（风险评分）：{pred_prob:.4f}")
        except:
            st.error("请输入所有指标的有效数字！")

# --------- 作品三：血压评估 ---------
def blood_pressure_app():
    st.title("作品三（血压评估）")
    st.write("请输入您的血压值进行评估:")

    sbp = st.number_input("收缩压 SBP (mmHg)", min_value=50, max_value=250, value=120)
    dbp = st.number_input("舒张压 DBP (mmHg)", min_value=30, max_value=150, value=80)

    if st.button("开始评估"):
        if sbp >= 140 or dbp >= 90:
            st.warning("血压偏高，建议咨询医生。")
        elif sbp < 90 or dbp < 60:
            st.warning("血压偏低，注意饮食和休息。")
        else:
            st.success("血压正常，请保持。")

# --------- 作品四：体重管理建议 ---------
def weight_management_app():
    st.title("作品四（体重管理建议）")
    st.write("请输入身高体重，我们将计算您的BMI并提供建议。")

    height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=65.0)

    if st.button("计算BMI"):
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        st.write(f"您的BMI是：{bmi:.2f}")
        if bmi < 18.5:
            st.info("偏瘦，建议增加营养摄入。")
        elif 18.5 <= bmi < 24:
            st.success("体重正常，继续保持。")
        elif 24 <= bmi < 28:
            st.warning("超重，建议运动饮食管理。")
        else:
            st.error("肥胖，建议积极干预。")

# --------- 首页 ---------
def home():
    st.title("首页 - 自我介绍")
    st.markdown("""
    ### 你好，我是张三
    欢迎使用我的健康评估平台。
    本平台包含多个小工具，帮你更好地了解健康状况。
    """)

# --------- 主程序入口 ---------
def main():
    st.sidebar.title("导航菜单")
    page = st.sidebar.radio("请选择页面", [
        "首页",
        "作品一（健康系统评估）",
        "作品二（糖尿病风险评估）",
        "作品三（血压评估）",
        "作品四（体重管理建议）"
    ])

    if page == "首页":
        home()
    elif page == "作品一（健康系统评估）":
        health_system_app()
    elif page == "作品二（糖尿病风险评估）":
        diabetes_model_app()
    elif page == "作品三（血压评估）":
        blood_pressure_app()
    elif page == "作品四（体重管理建议）":
        weight_management_app()

if __name__ == "__main__":
    main()
