#1 Khởi tạo và nạp mô hình 

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Nạp file mô hình đã lưu ở Bước 2
model = joblib.load("svm_model.pkl")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)
#2 Khai báo cấu trúc dữ liệu và mapping nhãn

# Lớp định nghĩa dữ liệu đầu vào sử dụng Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Từ điển chuyển đổi từ class_id sang tên loài hoa
species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}
#3 Tạo các Endpoint (Đường dẫn xử lý)
# Endpoint kiểm tra API có đang chạy không
@app.get("/")
def home():
    return {"message": "Iris SVM API is running"}

# Endpoint health check (quan trọng khi deploy)
@app.get("/health")
def health():
    return {"status": "healthy"}

# Endpoint chính để nhận dữ liệu và trả về dự đoán
@app.post("/predict")
def predict(data: IrisInput):
    # Trích xuất dữ liệu truyền vào thành mảng 2 chiều
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    
    # Đưa vào mô hình dự đoán
    prediction = int(model.predict(features)[0])
    
    # Trả về kết quả JSON
    return {
        "class_id": prediction,
        "prediction": species[prediction],
    }