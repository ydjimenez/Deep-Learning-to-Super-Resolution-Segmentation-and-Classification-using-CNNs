import h5py
import h5py  # Kit de herramientas de importación
import numpy as np

with h5py.File('nuev.h5', "r") as f:
    for key in f.keys():
        # print (f [clave], clave, f [clave] .name, f [clave] .value) # Debido a que hay un objeto de grupo que no tiene atributo de valor, será anormal. Además, la cadena de caracteres se lee como una secuencia de bytes y debe decodificarse en una cadena de caracteres.
        print(f[key], key, f[key].name)


print("Segundo bloque de codigo ")
f = h5py.File('nuev.h5', 'r')
a = f["hr"].name[:] 
 # Recuperar todos los valores clave cuya clave principal es la imagen
print(a)
f.close()

with h5py.File('nuev.h5', "r") as f:
	for key in f.keys():
		a = f["lr"].name[:] 
		# Recuperar todos los valores clave cuya clave principal es la imagen
		print(a)



