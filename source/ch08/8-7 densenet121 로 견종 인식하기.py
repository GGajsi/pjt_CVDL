
import pathlib
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Rescaling,
    Normalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory


# ============================================
# 1. 데이터셋 경로 설정
# ============================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'datasets' / 'stanford_dogs' / 'Images'
SAVE_DIR = pathlib.Path(__file__).resolve().parent

print("프로젝트 경로:", PROJECT_ROOT)
print("데이터셋 경로:", DATA_PATH)
print("데이터셋 존재:", DATA_PATH.is_dir())

if not DATA_PATH.is_dir():
    raise FileNotFoundError(f"데이터셋 폴더 없음: {DATA_PATH}")


# ============================================
# 2. 데이터셋 불러오기
# ============================================

train_ds = image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

val_ds = image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

class_names = train_ds.class_names

print("견종 개수:", len(class_names))
print("훈련 배치 수:", len(train_ds))
print("검증 배치 수:", len(val_ds))


# ============================================
# 3. DenseNet121 사전 학습 모델
# ============================================

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 기존 가중치를 고정하여 학습 시간을 줄임
base_model.trainable = False


# ============================================
# 4. CNN 모델 구성
# ============================================

cnn = Sequential([
    Input(shape=(224, 224, 3)),

    # DenseNet121 전용 전처리
    Rescaling(1.0 / 255.0),

    Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    ),

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(120, activation='softmax')
])


# ============================================
# 5. 모델 컴파일
# ============================================

cnn.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

cnn.summary()


# ============================================
# 6. 모델 학습
# ============================================

EPOCHS = 1  # 처음에는 1로 실행, 이후 3~5로 변경

hist = cnn.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=2
)


# ============================================
# 7. 검증 정확도 확인
# ============================================

loss, accuracy = cnn.evaluate(
    val_ds,
    verbose=0
)

print(f"\n검증 손실: {loss:.4f}")
print(f"검증 정확도: {accuracy * 100:.2f}%")


# ============================================
# 8. 학습 모델 저장
# ============================================

model_path = SAVE_DIR / 'cnn_for_stanford_dogs.keras'

cnn.save(str(model_path))

print("모델 저장 완료:", model_path)


# ============================================
# 9. 견종 이름 저장
# ============================================

names_path = SAVE_DIR / 'dog_species_names.pkl'

with open(names_path, 'wb') as f:
    pickle.dump(class_names, f)

print("견종 이름 저장 완료:", names_path)


# ============================================
# 10. 정확도 그래프
# ============================================

plt.figure(figsize=(9, 4))

plt.plot(
    hist.history['accuracy'],
    marker='o',
    label='Train'
)

plt.plot(
    hist.history['val_accuracy'],
    marker='o',
    label='Validation'
)

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


# ============================================
# 11. 손실 그래프
# ============================================

plt.figure(figsize=(9, 4))

plt.plot(
    hist.history['loss'],
    marker='o',
    label='Train'
)

plt.plot(
    hist.history['val_loss'],
    marker='o',
    label='Validation'
)

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
