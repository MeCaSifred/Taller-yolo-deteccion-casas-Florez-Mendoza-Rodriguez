from ultralytics import YOLO
import os, shutil, glob

def entrenar():
    print("\n Iniciando entrenamiento...")

    model = YOLO('yolov8n.pt')  # se descarga automático la primera vez
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        optimizer='AdamW',
        lr0=0.001,
        warmup_epochs=5,
        flipud=0.0,
        fliplr=0.5,
        degrees=15,
        scale=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        project='runs',
        name='casas'
    )

    # Guardar pesos en models/
import glob
rutas = glob.glob('runs/**/*.pt', recursive=True)
rutas = [r for r in rutas if 'best.pt' in r]
if rutas:
    mejor = max(rutas, key=os.path.getmtime)
    os.makedirs('models', exist_ok=True)
    shutil.copy(mejor, 'models/house-yolo.pt')
    print(f"\n Pesos copiados desde: {mejor}")
    print(" Guardados en: models/house-yolo.pt")
else:
    print("\n No se encontró best.pt")

if __name__ == "__main__":
    entrenar()