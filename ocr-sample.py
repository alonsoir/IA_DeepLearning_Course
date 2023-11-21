import pytesseract
from PIL import Image

from os import environ

# Establecer la variable de entorno TESSDATA_PREFIX
environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata'
try:

    text = pytesseract.image_to_string(Image.open("texto.png"))
    print(text)
    text1 = pytesseract.image_to_string(Image.open("texto1.png"))
    print(text1)
    text3 = pytesseract.image_to_string(Image.open("texto3.png"))
    print(text3)

    _config = r'--psm 10 --oem 3 --tessdata-dir "/usr/local/share/tessdata"'
    print("This should be ancient greek: ", pytesseract.image_to_string(Image.open("greek.jpg"), lang="grc", config=_config))

except Exception as ex:
    print(str(ex))
finally:
    print("done!")