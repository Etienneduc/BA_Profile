import streamlit as st
import pandas as pd
import pickle

# Load mô hình và encoder
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Trang trí giao diện
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.markdown("<h1 style='text-align: center; color: #009999;'>🔮 Dự đoán Khách hàng Rời bỏ (Churn)</h1>", unsafe_allow_html=True)
st.markdown("---")

# Chế độ lựa chọn
mode = st.radio("Chọn chế độ dự đoán:", ["🔹 Nhập từng khách hàng", "📁 Tải file CSV nhiều khách hàng"])

if mode == "🔹 Nhập từng khách hàng":
    st.subheader("📌 Nhập thông tin khách hàng:")

    gender = st.selectbox("Giới tính", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Khách hàng cao tuổi?", [0, 1])
    Partner = st.selectbox("Có vợ/chồng?", ["Yes", "No"])
    Dependents = st.selectbox("Có người phụ thuộc?", ["Yes", "No"])
    tenure = st.number_input("Số tháng gắn bó (tenure)", min_value=0, max_value=100)
    PhoneService = st.selectbox("Sử dụng dịch vụ điện thoại?", ["Yes", "No"])
    MultipleLines = st.selectbox("Sử dụng nhiều đường dây?", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Loại Internet", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Bảo mật online?", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Sao lưu online?", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Bảo vệ thiết bị?", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Hỗ trợ kỹ thuật?", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Xem TV trực tuyến?", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Xem phim trực tuyến?", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Hóa đơn điện tử?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Phương thức thanh toán", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Phí hàng tháng", min_value=0.0)
    TotalCharges = st.number_input("Tổng phí đã trả", min_value=0.0)

    input_data = pd.DataFrame([{ "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges }])

    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[col])

    if st.button("📈 Dự đoán"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Kết quả dự đoán:")
        st.metric("Xác suất khách hàng rời bỏ", f"{prob:.2%}")
        st.progress(prob)

        if prediction == 1:
            st.error("❌ Kết luận: Khách hàng CÓ khả năng rời bỏ dịch vụ.")
        else:
            st.success("✅ Kết luận: Khách hàng KHÔNG có khả năng rời bỏ dịch vụ.")

else:
    st.subheader("📁 Tải file CSV chứa nhiều khách hàng")
    uploaded_file = st.file_uploader("Tải file .csv", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 Dữ liệu gốc:")
        st.dataframe(data.head())

        for col, encoder in encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col])

        predictions = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        data["Churn Prediction"] = ["Churn" if p == 1 else "No Churn" for p in predictions]
        data["Churn Probability"] = (probs * 100).round(2).astype(str) + "%"

        st.subheader("📈 Kết quả dự đoán:")
        st.dataframe(data[["Churn Prediction", "Churn Probability"]])

        # Tính toán và hiển thị tỷ lệ churn
        total = len(data)
        churn_count = (data["Churn Prediction"] == "Churn").sum()
        no_churn_count = total - churn_count

        churn_rate = churn_count / total * 100
        no_churn_rate = no_churn_count / total * 100

        st.subheader("📊 Tỷ lệ churn trong tập dữ liệu:")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("💔 Rời bỏ (Churn)", f"{churn_rate:.2f}%", delta=f"{churn_count} khách hàng")
        with col2:
            st.metric("💚 Ở lại (No Churn)", f"{no_churn_rate:.2f}%", delta=f"{no_churn_count} khách hàng")

        st.success(f"✅ Đã xử lý {total} khách hàng.")
