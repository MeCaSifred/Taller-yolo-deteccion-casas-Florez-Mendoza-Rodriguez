from ultralytics import YOLO
import os

def detectar(ruta_imagen, ruta_pesos='models/house-yolo.pt'):
    if not os.path.exists(ruta_pesos):
        print(f" Modelo no encontrado: {ruta_pesos}")
        return
    if not os.path.exists(ruta_imagen):
        print(f" Imagen no encontrada: {ruta_imagen}")
        return

    model = YOLO(ruta_pesos)
    results = model(ruta_imagen)

    print(f"\n Casas detectadas: {len(results[0].boxes)}")
    for i, box in enumerate(results[0].boxes):
        print(f"  Casa {i+1} | Confianza: {float(box.conf[0]):.2f}")

    results[0].save(filename='resultado.jpg')
    print(" Guardado en resultado.jpg")

if __name__ == "__main__":
    import sys
    imagen = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    detectar(imagen)