import os
import pandas as pd
import cv2
import deepface
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from deepface import DeepFace
import deepface
import shutil
import time

def getEmotions(objs):
    for obj in objs:
        # Extraer los valores dominantes y convertirlos en una cadena
        dominant_gender = obj.get("dominant_gender", "")
        dominant_emotion = obj.get("dominant_emotion", "")
        dominant_race = obj.get("dominant_race", "")

        # Concatenar los valores en una cadena
        result_string = f"Gender: {dominant_gender}, Emotion: {dominant_emotion}, Race: {dominant_race}"

        # Imprimir o hacer lo que necesites con la cadena resultante
        print(result_string)
        return  result_string
def deepFind():
    # Crea una instancia de la clase DeepFace
    face_detector = DeepFace
    # Inicializa la cámara nuevamente
    cap = cv2.VideoCapture(0)
    # Obtén la resolución de la cámara
    resolution = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Establece la resolución de la ventana a 2K
    cv2.namedWindow("Capturadora", cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Capturadora", cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.resizeWindow("Cámara", 2048, 1024)
    # Directorio para almacenar temporalmente los fotogramas
    temp_dir = "temp_frames"
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Configura la grabación del video
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (video_width, video_height))

    # Bucle infinito para la detección en tiempo real
    while True:
        ret, frame = cap.read()
        if ret:
            # Convierte el fotograma a un formato de color adecuado
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            # Detecta los rostros en el fotograma
            # 1.1, 10: caras más pequeñas o parcialmente ocultas
            # 2.0, 2: caras más grandes.
            faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(frame, 1.3, 5)
            if len(faces) > 0:
                # Obtén el primer rostro detectado
                x, y, w, h = faces[0]
                # Recorta la región del rostro
                face_roi = frame[y:y + h, x:x + w]
                # Guarda el recorte como una imagen temporal
                # temp_path = os.path.join(temp_dir, "temp_face" + str(num_frames) + ".jpg")
                temp_path = os.path.join(temp_dir, "temp_face.jpg")
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA))

                listOfMatches=DeepFace.find(temp_path,db_path=_db_path,
                            model_name=_model_name,
                            distance_metric=_metrics,
                            enforce_detection=False)
                objs = DeepFace.analyze(img_path=temp_path,
                                        actions=['age', 'gender', 'race', 'emotion']
                                        )
                analysis = getEmotions(objs)
                # Concatena todos los DataFrames en uno solo
                merged_df = pd.concat(listOfMatches, ignore_index=True)

                # Filtra el DataFrame para obtener las coincidencias con un umbral de similitud específico
                umbral_similitud = _umbral_similitud  # ajusta según tus necesidades
                coincidencias_superiores = merged_df[merged_df['Facenet_cosine'] >= umbral_similitud]

                # Imprime las coincidencias que superan el umbral y sus valores de 'Facenet_cosine'
                for row in coincidencias_superiores.itertuples():
                    nombre = os.path.split(os.path.dirname(f"{row.identity}"))[-1]
                    print(f"Facenet_cosine: {row.Facenet_cosine}, Ruta de la imagen: {row.identity}, nombre: ", nombre)
                    if not coincidencias_superiores.empty:
                        for (x, y, w, h) in faces:
                            # Dibuja un cuadro sobre el rostro
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # Escribe el texto "ADMIN" sobre el cuadro
                            cv2.putText(frame, nombre, (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, analysis, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # Muestra el fotograma
                            cv2.imshow("Capturadora", frame)
                    else:
                        print("La persona en la imagen no está en la base de datos.")
            # Agrega el frame al video
        out.write(frame)
        # shutil.rmtree(temp_dir, ignore_errors=True)
        # os.makedirs(temp_dir, exist_ok=True)
        # print("borrado frame...")
        # time.sleep(1)
        # Espera a que se presione una tecla
        key = cv2.waitKey(1)

        # Si se presiona la tecla q, sale del bucle
        if key == ord("q"):
            break


    # Cierra la cámara
    cap.release()
    out.release()
    # Destruye todas las ventanas abiertas
    cv2.destroyAllWindows()

if __name__ == "__main__":
    _db_path = "/Users/aironman/git/users/database"
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ]
    _model_name = models[1]
    _umbral_similitud = 0.39
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    _metrics = metrics[0]
    try:
        print(_db_path,_model_name,_metrics)

        device = "/device:GPU:0"
        # isGPUAvailable = tf.test.is_gpu_available()
        isGPUAvailable = tf.config.list_physical_devices('GPU')

        if len(isGPUAvailable) == 0:
            print("Parece que la GPU no está disponible...")
            devices = tf.config.list_physical_devices(device)
            if len(devices) == 0:
                print("Tensorflow sólo podrá usar la CPU. A día de hoy, sólo NVIDIA te da soporte físico.")
        else:
            print("Parece que hay una GPU que podrías usar.")
        with tf.device(device):
            deepFind()
    except OSError as oSError:
        print(type(oSError))
    except AttributeError as attributeError:
        print(attributeError.args[0])
        print(type(attributeError))
    except KeyError as keyError:
        print(keyError.args[0])
        print(type(keyError))
    except ValueError as valueError:
        print(valueError.args[0])
        print(type(valueError))
    except TypeError as typeError:
        print(typeError.args[0])
        print(type(typeError))
    except Exception as unknownException:
        print(unknownException.args[0])
        print(type(unknownException))
    finally:
        print("Done!")
