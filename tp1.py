#!/usr/bin/env python3.9
from typing import Tuple, List
from utils.svg_plots import parse_svg_path_commands
from itertools import chain
from utils.misc import resample
import numpy as np
ichain = chain.from_iterable

def punto4():
    origen = (36.997, 31.431)
    escala = (.3/(375.9-36.997), 250/(261.37-31.431))
    segmentos_svg_respuesta = (
        "m 36.997,31.431 1.428,0.713 1.427,0.712 1.428,1.425 1.428,2.137 1.428,2.137 1.427,2.874 2.142,2.849 1.452,2.85 1.428,3.586 1.428,3.562 1.427,3.562 1.428,3.586 1.428,3.562 2.141,4.274 1.428,4.299 1.428,3.561 1.427,4.299 1.453,4.274 1.427,3.586 1.428,4.275 2.142,3.561 1.427,4.295 1.428,3.57 1.428,4.29 1.427,3.57 1.428,4.27 1.428,3.59 2.166,3.56 1.428,3.56 1.427,3.59 1.428,2.85 1.428,3.56 1.427,3.58 1.428,2.85 2.142,2.85 1.427,3.59 1.428,2.85 1.452,2.85 1.428,2.87 1.428,2.85 1.427,2.14 2.14,2.85 1.43,2.87 1.43,2.14 1.43,2.14 1.42,2.85 1.43,2.16 1.45,2.13 2.14,2.14 1.43,2.14 1.43,1.42 1.43,2.16 1.43,2.14 1.42,1.43 1.43,2.13 2.14,1.43 1.43,1.42 1.45,2.16 1.43,1.43 1.43,1.42 1.43,1.43 1.42,1.42 2.14,1.43 1.43,1.45 1.43,1.42 1.43,0.71 1.43,1.43 1.45,1.42 1.43,0.72 2.14,1.42 1.42,0.71 1.43,1.43 1.43,0.73 1.43,1.43 1.43,0.71 2.14,0.71 1.42,0.72 1.46,1.42 1.42,0.71 1.43,0.72 1.43,0.71 1.43,0.71 2.14,0.71 1.43,0.72 1.42,0.73 1.43,0.71 1.43,0.72 h 1.45 l 1.43,0.71 2.14,0.71 1.43,0.71 1.43,0.72 h 1.42 l 1.43,0.71 1.43,0.71 h 1.43 l 2.16,0.71 1.43,0.72 h 1.43 l 1.43,0.71 h 1.42 l 1.43,0.71 h 1.43 l 2.14,0.71 h 1.43 l 1.43,0.74 h 1.45 l 1.43,0.71 h 1.42 l 1.43,0.71 h 2.14 1.43 l 1.43,0.72 h 1.43 l 1.42,0.71 h 1.43 1.45 l 2.15,0.71 h 1.42 1.43 1.43 l 1.43,0.71 h 1.42 1.43 2.14 l 1.43,0.72 h 1.45",
        "m 231.48,243.52 h 1.43 1.43 l 1.43,0.71 h 1.42 2.15 1.42 1.43 l 1.43,0.71 h 1.43 1.45 1.43 2.14 1.43 l 1.42,0.71 h 1.43 1.43 1.43 1.43 2.14 1.45 1.43 l 1.42,0.72 h 1.43 1.43 2.14 1.43 1.43 1.42 1.43 1.45 1.43 2.14 l 1.43,0.71 h 1.43 1.43 1.42 1.43 1.43 2.17 1.42 1.43 1.43 1.43 1.42 1.43 2.14 1.43 1.43 l 1.43,0.71 h 1.45 1.43 1.42 2.15 1.42 1.43 1.43 1.43 1.42 1.43 2.17 1.43 1.42 1.43 1.43 1.43 1.42 2.15 1.42 1.46 1.42 1.43 1.43 1.43 2.14 1.43 1.42 1.43 1.43 1.45 1.43 2.14 1.43 1.43 1.42 1.43 1.43 l 1.43,0.74 h 2.14 1.45 1.43 1.43 1.42 1.43 2.14"
    )

    segmentos_svg_entrada = (
        "m 36.997,252.8 h 1.428 1.427 1.428 1.428 1.428 1.427 2.142 1.452 1.428 1.428 1.427 1.428 1.428 2.141 1.428 1.428 1.427 1.453 1.427 1.428 2.142 1.427 1.428 1.428 1.427 1.428 1.428 2.166 1.428 1.427 1.428 1.428 1.427 1.428 2.142 1.427 1.428 1.452 1.428 1.428 1.427 2.14 1.43 1.43 1.43 1.42 1.43 1.45 2.14 1.43 1.43 1.43 1.43 1.42 1.43 2.14 1.43 1.45 1.43 1.43 1.43 1.42 2.14 1.43 1.43 1.43 1.43 1.45 1.43 2.14 1.42 1.43 1.43 1.43 1.43 2.14 1.42 1.46 1.42 1.43 1.43 1.43 2.14 1.43 1.42 1.43 1.43 1.45 1.43 2.14 1.43 1.43 1.42 1.43 1.43 1.43 2.16 1.43 1.43 1.43 1.42 1.43 1.43 2.14 1.43 1.43 1.45 1.43 1.42 1.43 2.14 1.43 1.43 1.43 1.42 1.43 1.45 2.15 1.42 1.43 1.43 1.43 1.42 1.43 2.14 1.43 1.45",
        "m 231.48,252.8 h 1.43 1.43 1.43 1.42 2.15 1.42 1.43 1.43 1.43 1.45 1.43 2.14 1.43 1.42 1.43 1.43 1.43 1.43 2.14 1.45 1.43 1.42 1.43 1.43 2.14 1.43 1.43 1.42 1.43 1.45 1.43 2.14 1.43 1.43 1.43 1.42 1.43 1.43 2.17 1.42 1.43 1.43 1.43 1.42 1.43 2.14 1.43 1.43 1.43 1.45 1.43 1.42 2.15 1.42 1.43 1.43 1.43 1.42 1.43 2.17 1.43 1.42 1.43 1.43 1.43 1.42 2.15 1.42 1.46 1.42 1.43 1.43 1.43 2.14 1.43 1.42 1.43 1.43 1.45 1.43 2.14 1.43 1.43 1.42 1.43 1.43 1.43 2.14 1.45 1.43 1.43 1.42 1.43 2.14"
    )

    svg_a_normalizado = normalizador(escala, origen)
    respuesta = csplit(svg_a_normalizado(segmentos_svg_respuesta))
    entrada = csplit(svg_a_normalizado(segmentos_svg_entrada))
    #print(svg_a_normalizado(segmentos_svg_respuesta))
    time, (entrada_y, respuesta_y) = resample(entrada, respuesta)
    return time, entrada_y, respuesta_y

def punto3():
    escala = ((18-5)/(363.28-60.776), (12-3)/(262.09-32.883))
    origen = (60.776-5*(escala[0]**(-1)),32.883-3*(escala[1]**(-1)))
    segmentos_respuesta = (
        "m 60.776,67.862 h 0.714 0.713 0.714 0.739 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.738 0.714 0.714 0.714 0.714 0.714 0.714 0.713 0.714 l 0.714,0.712 1.428,1.425 v 3.586 l 0.714,0.712 v 4.987 l 0.713,0.712 v 6.436 l 0.714,0.712 v 7.148 l 0.714,0.713 v 8.545 l 0.714,0.74 v 10 l 0.714,0.71 v 10.71 l 0.714,0.71 v 10.71 l 0.714,0.71 v 11.43 l 0.738,0.71 v 11.42 l 0.714,0.71 v 10.71 l 0.714,0.72 v 11.42 l 0.714,0.71 v 10 l 0.713,0.71 v 10 l 0.714,0.71 v 8.57 l 0.714,0.71 v 8.58 l 0.714,0.71 v 6.44 l 0.714,0.71 v 5.7 l 0.714,0.71 v 4.3 l 0.714,0.71 v 3.56 l 0.713,0.71 v 2.14 l 0.713,0.74 h 0.72 l 1.42,-1.45 v -2.14 l 0.72,-0.71 v -2.85 l 0.71,-0.71 v -4.3 l 0.71,-0.71 v -4.28 l 0.74,-0.71 v -5.72 l 0.72,-0.72 v -5.69 l 0.71,-0.72 v -7.14 l 0.71,-0.72 v -6.43 l 0.72,-0.72 v -7.12 l 0.71,-0.74 v -7.12 l 0.72,-0.71 v -7.15 l 0.71,-0.71 v -7.15 l 0.71,-0.71 v -6.41 l 0.72,-0.72 v -5.72 l 0.71,-0.71 v -5.72 l 0.72,-0.72 v -4.98 l 0.71,-0.72 v -3.58 l 0.71,-0.71 v -3.57 l 0.72,-0.71 v -2.85 l 0.71,-0.71 v -1.45 l 1.43,-1.42 v -0.72 h 0.71 l 0.72,0.72 0.73,0.71 v 1.45 l 0.72,0.71 v 1.42 l 0.71,0.72 v 2.13 l 0.72,0.72",
        "m 121.58,141.41 v 2.85 l 0.71,0.71 v 3.58 l 0.71,0.72 v 3.56 l 0.72,0.71 v 4.3 l 0.71,0.71 v 4.28 l 0.72,0.71 v 4.3 l 0.71,0.71 v 4.27 l 0.71,0.72 v 4.29 l 0.72,0.72 v 4.27 l 0.71,0.71 v 3.56 l 0.71,0.72 v 4.3 l 0.72,0.71 v 2.85 l 0.71,0.71 v 2.87 l 0.72,0.72 v 2.13 l 0.71,0.72 v 2.13 l 0.71,0.71 v 1.43 l 1.46,1.42 v 0.74 l 0.71,0.71 h 0.71 l 0.72,-0.71 1.43,-1.45 v -1.42 l 0.71,-0.72 v -1.42 l 0.71,-0.71 v -1.43 l 0.72,-0.71 v -2.14 l 0.71,-0.71 v -2.16 l 0.71,-0.71 v -2.14 l 0.72,-0.71 v -2.85 l 0.71,-0.71 v -2.88 l 0.72,-0.71 v -2.14 l 0.71,-0.71 v -2.85 l 0.71,-0.71 v -2.16 l 0.72,-0.72 v -2.13 l 0.71,-0.71 v -2.14 l 0.72,-0.71 v -2.14 l 0.73,-0.71 v -1.45 l 0.72,-0.72 v -1.42 l 1.43,-1.42 v -1.43 l 0.71,-0.71 0.71,-0.71 h 0.72 0.71 l 0.71,0.71 0.72,0.71 0.71,0.71 1.43,1.43 v 1.42 l 0.71,0.71 v 1.45 l 1.43,1.43 v 2.14 l 0.72,0.71 v 1.42 l 0.71,0.72 v 1.42 l 0.71,0.71 v 1.43 l 0.72,0.73 v 1.43 l 0.74,0.71 v 1.43 l 0.71,0.71 v 1.42 l 1.43,1.43 v 1.42 l 1.42,1.43 v 1.45 l 1.43,1.42 v 0.71 h 0.72 l 0.71,0.72 h 0.71 0.72 0.71 l 0.72,-0.72 0.71,-0.71 0.71,-0.71 1.43,-1.43 v -0.73 l 1.45,-1.43 v -1.42 l 1.43,-1.43 v -1.42 l 1.43,-1.43 v -1.42 l 0.71,-0.71 1.43,-1.45 v -1.43 l 0.71,-0.71 1.43,-1.42 v -0.72 l 0.72,-0.71 h 0.71 l 0.71,-0.71 h 0.72",
        "m 185.21,171.4 h 0.71 0.71 0.72 l 0.71,0.71 h 0.72 l 0.73,0.71 0.72,0.72 0.71,0.71 0.72,0.71 1.42,1.42 v 0.72 l 0.72,0.73 0.71,0.72 0.72,0.71 1.42,1.42 v 0.72 l 0.72,0.71 0.71,0.71 0.71,0.71 h 0.72 l 0.71,0.72 h 0.72 l 0.71,0.71 h 0.71 0.72 0.74 l 0.71,-0.71 h 0.71 0.72 l 0.71,-0.72 h 0.72 l 0.71,-0.71 0.71,-0.71 0.72,-0.71 h 0.71 l 0.71,-0.72 0.72,-0.71 0.71,-0.71 0.72,-0.71 h 0.71 l 0.71,-0.72 0.72,-0.73 h 0.71 l 0.72,-0.72 h 0.73 0.72 0.71 0.72 l 0.71,-0.71 0.71,0.71 h 0.72 0.71 0.71 l 0.72,0.72 h 0.71 0.72 l 0.71,0.73 h 0.71 l 0.72,0.72 h 0.71 l 0.72,0.71 0.71,0.71 h 0.71 0.72 l 0.74,0.71 h 0.71 l 0.71,0.72 h 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 l 0.71,-0.72 h 0.71 0.72 0.71 l 0.71,-0.71 h 0.74 0.72 l 0.71,-0.71 h 0.71 0.72 l 0.71,-0.71 h 0.72 0.71 0.71 0.72 l 0.71,-0.72 h 0.72 0.71 0.71 0.72 0.71 l 0.72,0.72 h 0.71 0.71 0.72 0.74 0.71 l 0.71,0.71 h 0.72 0.71 0.71 0.72 l 0.71,0.71 h 0.72 0.71 0.71 0.72 0.71 l 0.72,0.71 h 0.71 0.71 0.72 0.71 0.71 0.72 0.74 l 0.71,-0.71 h 0.71 0.72",
        "m 276.04,179.97 h 0.71 0.72 0.71 0.71 0.72 l 0.71,-0.71 h 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 l 0.73,-0.71 h 0.72 0.71 l 0.72,0.71 h 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 l 0.71,0.71 h 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 l 0.72,-0.71 h 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 l 0.71,0.71 h 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.74 l 0.71,-0.71 h 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71"
    )

    segmentos_entrada = (
        "m 60.776,58.576 h 0.714 0.713 0.714 0.739 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.738 0.714 0.714 0.714 0.714 0.714 0.714 0.713 0.714 0.714 V 160.69 h 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.714 0.738 0.714 0.714 0.714 0.713 0.714 0.714 0.714 0.714 0.714 0.714 0.713 0.713 0.72 0.71 0.71 0.72 0.71 0.71 0.74 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71",
        "m 150.89,160.69 h 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.74 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71",
        "m 241.7,160.69 h 0.71 0.72 0.71 0.71 0.74 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71",
        "m 332.53,160.69 h 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.74 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.73 0.72 0.71 0.72 0.71 0.71 0.72 0.71 0.72 0.71"
    )
    svg_a_normalizado = normalizador(escala, origen)
    respuesta = csplit(svg_a_normalizado(segmentos_respuesta))
    entrada = csplit(svg_a_normalizado(segmentos_entrada))

    time, (entrada_y, respuesta_y) = resample(entrada, respuesta)
    return time, entrada_y, respuesta_y


csplit = lambda two_col_array: (two_col_array[:,0], two_col_array[:,1])

def normalizador(escala, origen):
    normalizar = lambda arr: (arr - origen) * np.array([escala])
    array_iter2d = lambda iter: np.array(list(iter))
    segmentos_svg = lambda segs: chain.from_iterable(map(parse_svg_path_commands, segs))
    return lambda segs: normalizar(array_iter2d(segmentos_svg(segs)))


if __name__ == "__main__":
    np.savetxt("tp1/punto4_datos.csv",
        np.column_stack(punto4()),
        delimiter=",",
        header="Tiempo [s],Tensión de armadura [V],Velocidad [rad/s]"
    )
    np.savetxt("tp1/punto3_datos.csv",
        np.column_stack(punto3()),
        delimiter=",",
        header="Tiempo [s],Tensión [V],Desplazamiento [mm]"
    )
