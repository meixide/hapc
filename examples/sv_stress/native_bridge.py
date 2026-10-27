import ctypes as ct
import numpy as np
from pathlib import Path

def fit(lib,y,z,v,a,logistic=0,steps=150,factor=.8):
    lib=ct.CDLL(str(lib)); f=lib.fit
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f.argtypes=[ct.c_int]*4+[ptr]*4+[ct.c_int,ct.c_double,ct.c_double,ptr,ct.POINTER(ct.c_double),ct.c_char_p]
    y,z,v,a=[np.ascontiguousarray(x,dtype=float) for x in (y,z,v,a)]
    hist=np.zeros((steps+1,len(a))); risk=ct.c_double(); status=ct.create_string_buffer(128)
    rows=f(logistic,len(y),len(a),len(v),y,z,v,a,steps,1e-10,factor,hist,ct.byref(risk),status)
    if rows<0: raise ValueError(status.value.decode())
    return hist[:rows],risk.value,status.value.decode()

