import time
import numpy as np

def sajat_szum(xs):
   szum = 0
   for v in xs:
       szum += v
   return szum

sz = 10000000        # Legyen 10 millió eleme a listának
testadat = range(sz)

t0 = time.perf_counter()
sajat_eredmeny = sajat_szum(testadat)
t1 = time.perf_counter()
print("saját_eredmény = {0} (eltelt ido = {1:.4f} másodperc)".format(sajat_eredmeny, t1-t0))

t2 = time.perf_counter()
gepi_eredmeny = sum(testadat)
t3 = time.perf_counter()
print("gépi_eredmény = {0} (eltelt idő = {1:.4f} másodperc)".format(gepi_eredmeny, t3-t2))

t4 = time.perf_counter()
np_eredmeny = np.sum(testadat)
t5 = time.perf_counter()
print("numpy_eredmény = {0} (eltelt idő = {1:.4f} másodperc)".format(np_eredmeny, t5-t4))

