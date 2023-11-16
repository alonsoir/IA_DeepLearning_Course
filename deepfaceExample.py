# Importa las bibliotecas necesarias

from deepface import DeepFace


def demo():
    try:
        result = DeepFace.verify(img1_path=_img1_path, img2_path=_img2_path)
        print("result is ", result)

        dfs = DeepFace.find(img_path="img1.jpeg", db_path=_db_path)
        print("dfs is ", dfs)

        embedding_objs = DeepFace.represent(img_path=_img1_path)
        embedding = embedding_objs[0]["embedding"]
        assert isinstance(embedding, list)

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

        # face verification
        result = DeepFace.verify(img1_path=_img1_path,
                                 img2_path=_img2_path,
                                 model_name=models[0]
                                 )
        print(result)
        # face recognition
        dfs = DeepFace.find(img_path="img1.jpeg",
                            db_path=_db_path,
                            model_name=models[1],
                            enforce_detection=False
                            )
        print(dfs)
        # face recognition
        dfs = DeepFace.find(img_path="img2.jpeg",
                            db_path=_db_path,
                            model_name=models[1],
                            enforce_detection=False
                            )
        print(dfs)
        # embeddings
        embedding_objs = DeepFace.represent(img_path=_img1_path,
                                            model_name=models[2]
                                            )
        metrics = ["cosine", "euclidean", "euclidean_l2"]

        # face verification
        result = DeepFace.verify(img1_path=_img1_path,
                                 img2_path=_img2_path,
                                 distance_metric=metrics[1]
                                 )
        print(result)
        # face recognition
        dfs = DeepFace.find(img_path=_img1_path,
                            db_path=_db_path,
                            distance_metric=metrics[2]
                            )
        print(dfs)
        objs = DeepFace.analyze(img_path=_img1_path,
                                actions=['age', 'gender', 'race', 'emotion']
                                )
        print(objs)
        backends = [
            'opencv',
            'ssd',
            'dlib',
            'mtcnn',
            'retinaface',
            'mediapipe',
            'yolov8',
            'yunet',
            'fastmtcnn',
        ]

        # face verification
        obj = DeepFace.verify(img1_path=_img1_path,
                              img2_path=_img2_path,
                              detector_backend=backends[0]
                              )
        print(obj)
        # face recognition
        dfs = DeepFace.find(img_path=_img1_path,
                            db_path=_db_path,
                            detector_backend=backends[1]
                            )
        print(dfs)
        # embeddings
        embedding_objs = DeepFace.represent(img_path=_img1_path,
                                            detector_backend=backends[2]
                                            )
        print(embedding_objs)
        # facial analysis
        demographies = DeepFace.analyze(img_path=_img1_path,
                                        detector_backend=backends[3]
                                        )
        print(demographies)
        # face detection and alignment
        face_objs = DeepFace.extract_faces(img_path=_img1_path,
                                           target_size=(224, 224),
                                           detector_backend=backends[4]
                                           )
        print(face_objs)


    except ValueError as valueError:
        print(str(valueError))
    except ModuleNotFoundError as moduleNotFoundError:
        print(str(moduleNotFoundError))
    except KeyError as keyError:
        print(str(keyError))
    except Exception as ex:
        print("Some generic error...")
        print(type(ex))


if __name__ == "__main__":
    _img1_path = "img1.jpeg"
    _img2_path = "img2.jpeg"
    _db_path = "/Users/aironman/git/users/database"

    demo()

    # Encuentra las caras en el fotograma
    faces = DeepFace.stream(db_path=_db_path)
    print("faces is ", faces)

    print("Done!")
