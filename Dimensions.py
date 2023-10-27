import json

"""
El proposito de este script es inicializar una estructura de datos que contenga un número determinado de constantes universales
163 cargadas a partir del fichero constantes_fundamentales.json, de manera en nuestra brana, la 3D, aparezcan las que hemos medido a día de hoy.

En el resto de dimensiones, he hecho un cálculo muy naive del 10% de los valores de nuestra brana.

El propósito es tener dicha estructura para luego tratar de encontrar algún mecanismo, alguna relación, o tratar de inferrir
algún comportamiento inusual según esos cambios en las constantes fundamentales.

Me gustaría añadir el resto de ellas, a ver si encuentro un json con el formato adecuado.
"""


def main():
    # Definimos las dimensiones del array
    num_dimensions = 11

    # Cargamos las constantes fundamentales
    with open("constantes_fundamentales.json", "r") as f:
        constantes = json.load(f)

    num_constantes = len(constantes)

    # Inicializamos el array
    array = [[{} for _ in range(num_constantes)] for _ in range(num_dimensions)]

    initialize_Dimensions(array, constantes, num_dimensions)

    show_constants(array, num_constantes, num_dimensions)


def show_constants(array, num_constantes, num_dimensions):
    # Mostramos todas las constantes por pantalla
    for i in range(num_dimensions):
        for j in range(num_constantes):
            constante = array[i][j]["constante"]
            definicion = array[i][j]["definicion"]
            unidad = array[i][j]["unidad"]
            valor = array[i][j]["valor"]

            if i < num_dimensions and j < num_constantes:
                print(
                    f"En la dimension {i} la constante {constante} tiene un valor de {valor}, definicion es {definicion}, unidad es {unidad}"
                )
            else:
                print(f"No hay una constante en la posición {i}:{j}")


def initialize_Dimensions(array, constantes, num_dimensions):
    num_constantes = len(constantes)
    # Guardamos las constantes en el array
    for i in range(num_constantes):
        array[2][i] = constantes[i]
        # Inicializamos el resto de dimensiones
    for i in range(num_dimensions):
        for j in range(num_constantes):
            if i != 2:
                constante = constantes[j]
                valor = constante["valor"] * 1.1
                definicion = constante["definicion"]
                unidad = constante["unidad"]
                """
                print("Dimension: ", i)
                print("Num constante: ", j)
                print("constante: ", constante)
                print("valor en la 3d: ", constante["valor"])
                print("valor alterado: ", valor)
                print("definicion:", definicion)
                print("unidad: ", unidad)
                """
                array[i][j] = {
                    "constante": constante["constante"],
                    "valor": valor,
                    "definicion": definicion,
                    "unidad": unidad,
                }
    print("Universo inicializado.")


if __name__ == "__main__":
    main()
