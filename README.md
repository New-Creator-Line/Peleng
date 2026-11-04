# Peleng 3D Multi-Camera Tracking Engine

Высокопроизводительный аппаратно-независимый программный комплекс для сквозной 3D-реконструкции и отслеживания траектории движения объектов по данным синхронизированной 5-камерной системы.

Проект разработан с использованием **C++20 (Eigen3, pybind11)** и **Python 3.14 (PyTorch, Ultralytics YOLOv8 / RT-DETR, OpenCV)**.

---

## 🚀 Быстрый старт

### 1. Запуск полного пайплайна (3D реконструкция + трекинг + метрики)
```bash
python main.py
```
> По умолчанию используются каноничные пути:
> - Калибровки камер: `data/raw/hackathon/`
> - Эталонная траектория: `data/ground_truth/object_transform_SM_AA_track.csv`
> - 2D детекции: `results/detections/detections_2d_yolo_real.csv`
> *(Для полной обратной совместимости аргументы `--calib-dir хакатон` и путь к старому расположению также полностью поддерживаются).*

### 2. Запуск в режиме слепого тестирования (Blind Test)
```bash
python main.py --calib-dir test_dataset --detections results/detections_test.csv --gt none --output results/trajectories/output_blind.csv
```

### 3. Запуск набора автотестов
```bash
python -m pytest tests/ -v
```

---

## 📁 Структура проекта

```
Peleng/
├── data/                               # Все входные данные проекта
│   ├── raw/hackathon/                  # Видеопотоки (MP4) и калибровки камер (CSV)
│   ├── ground_truth/                   # Эталонные траектории
│   ├── datasets/yolo_v3/               # Обучающая выборка YOLO (train/valid/data.yaml)
│   └── assets/                         # 3D-модели (CargoTruck.fbx)
├── weights/                            # Веса нейросетевых моделей
│   ├── pretrained/                     # Базовые веса (yolov8x.pt, rtdetr-l.pt)
│   └── finetuned/                      # Дообученные веса (peleng_yolov8x_best.pt, peleng_rtdetr_best.pt)
├── scripts/                            # Исполняемые утилиты
│   ├── training/                       # Скрипты обучения (train_yolo.py, train_rtdetr.py)
│   ├── inference/                      # Инференс по 5 камерам (process_videos_yolo.py)
│   └── dataset/                        # Сплит и валидация данных (split_v3_dataset.py)
├── src/                                # Алгоритмическое ядро
│   ├── peleng_core/                    # C++20 ускорение (Eigen3 + pybind11)
│   │   ├── camera_model.hpp            # Преобразования систем координат UE5 -> OpenCV
│   │   ├── triangulator.hpp            # Многокамерная DLT триангуляция (Jacobi SVD)
│   │   ├── kalman_filter.hpp           # 6-параметрический 3D фильтр Калмана
│   │   └── bindings.cpp                # Модуль привязки к Python
│   ├── camera_model.py                 # Модель виртуальных камер
│   ├── triangulator.py                 # Python триангуляция с автопереключением на C++
│   ├── tracker_3d.py                   # 3D фильтрация с компенсацией окклюзий
│   ├── yolo_detector.py                # Детекторы YOLOv8x / RT-DETR с анализом окклюзий
│   ├── evaluator.py                    # Расчет 3D MAE/RMSE + исправление сбоя дат Excel
│   ├── visualizer.py                   # 3D построение траекторий
│   └── mock_detections.py              # Генерация синтетических детекций для тестов
├── results/                            # Выходные файлы
│   ├── detections/                     # 2D детекции (CSV)
│   ├── trajectories/                   # Восстановленные 3D координаты (CSV)
│   ├── metrics/                        # JSON сводки точности и времени
│   ├── plots/                          # Графики сравнения с Ground Truth (PNG)
│   └── videos/                         # Обработанные видеопотоки и 6-панельный дашборд
├── docs/                               # Документация и технические условия
│   └── specifications/                 # ТЗ хакатона (TZ_hackathon.pdf)
├── tests/                              # Набор unit и integration тестов
├── main.py                             # Основная точка входа
└── хакатон/                            # Папка для совместимости со старыми скриптами
```

---

## 🛠️ Сборка нативного модуля C++20 (`peleng_core`)

Модуль C++20 обеспечивает скорость свыше **8 000+ FPS** при обработке многокамерных лучей.

Сборка и установка в окружение:
```bash
pip install -e . --no-build-isolation
```
Или прямая сборка с помощью setuptools:
```bash
python setup.py build_ext --inplace
```

---

## 🎯 Обучение и инференс детекторов

### Инференс по 5 видеокамерам и построение мульти-дашборда:
```bash
python scripts/inference/process_videos_yolo.py
```

### Дообучение YOLOv8x:
```bash
python scripts/training/train_yolo.py --epochs 50 --batch 8 --imgsz 512 --device 0
```

### Обучение RT-DETR:
```bash
python scripts/training/train_rtdetr.py --epochs 35 --batch 4 --imgsz 640 --device 0
```
