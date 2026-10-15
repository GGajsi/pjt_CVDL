import pickle
from pathlib import Path

# 현재 Python 파일과 같은 폴더의 pkl 파일
file_path = Path(__file__).resolve().parent / 'dog_species_names.pkl'

# pickle 파일 읽기
with open(file_path, 'rb') as f:
    dog_species = pickle.load(f)

# 내용 출력
print("데이터 타입:", type(dog_species))
print("전체 견종 수:", len(dog_species))

print("\n===== 견종 목록 =====")

for i, name in enumerate(dog_species, start=1):
    print(f"{i:3d}. {name}")