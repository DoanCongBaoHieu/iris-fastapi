from sklearn import datasets
from sklearn.svm import SVC
import joblib

# Tải tập dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Khởi tạo và huấn luyện mô hình SVM với kernel tuyến tính
model = SVC(kernel="linear")
model.fit(X, y)

# Lưu mô hình đã huấn luyện bằng joblib
joblib.dump(model, "svm_model.pkl")
print("Model saved!")