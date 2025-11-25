import astropy.io.fits
import pylab
import numpy as np
import sys

f1 = astropy.io.fits.open("jacobian_2023-06-09_paplc_640nm.fits")
f5 = astropy.io.fits.open("estimator0_efield_640nm.fits")

E0_real_full = f5[0].data[0].ravel()
E0_imag_full = f5[0].data[1].ravel()

DH_pixels = E0_real_full**2 + E0_imag_full**2 > 0
DH = np.where(DH_pixels)

J_real_full = f1[1].data.reshape((-1,320,160))[:,:160].reshape((-1,160*160))
J_imag_full = f1[1].data.reshape((-1,320,160))[:,160:].reshape((-1,160*160))
print(J_real_full)

E0_real_DH = E0_real_full.ravel()[DH]
E0_imag_DH = E0_real_full.ravel()[DH]
J_real_DH = J_real_full.T[DH]
J_imag_DH = J_imag_full.T[DH]


r = J_real_DH.shape[1]

G = np.stack([J_real_DH, J_imag_DH], axis=1)
E0 = np.stack([E0_real_DH, E0_imag_DH], axis=1).reshape((-1,1,2))

N = G.shape[0]
c = 1

Q = float(sys.argv[1])**2*np.eye(r)
D = 0
SNR = float(sys.argv[2])
flux = SNR**2/np.mean(E0_real_DH**2 + E0_imag_DH**2)

print(flux)

#all values are real
#G - sensitivity matrix of shape (N,2*c,r)
#N - number of pixels
#c - number of channels in one pixel (c=1) for monochromatic light or IFS
#r - number of modes
#Q - covariance of WFE modes (per exposure) of shape (r,r)
#E0 - static electric field of shape (N,1,2*c)
#flux - photon flux (per exposure) (see scaling of intensity at the bottom)
#D - dark current per pixel etc

P = Q*0.0
contrasts = []

#Iterations of ALGORITHM 1
for _ in range(64):
    eps = np.random.multivariate_normal(np.zeros(r), P+Q).reshape((1,1,r)) #random modes

    G_eps = np.sum(G*eps, axis=2).reshape((N,1,2*c)) + E0 #electric field
    G_eps_squared = np.sum(G_eps*G_eps, axis=2, keepdims=True)
    G_eps_G = np.matmul(G_eps, G)
    G_eps_G_scaled = G_eps_G/np.sqrt(G_eps_squared + D/flux) #trick to save RAM

    cal_I = 4*flux*np.einsum("ijk,ijl->kl", G_eps_G_scaled, G_eps_G_scaled) #information matrix
    P = np.linalg.inv(np.linalg.inv(P+Q) + cal_I)

    intensity = G_eps_squared*flux + D
    contrasts.append(np.mean(intensity)/flux)
    print("est. contrast", np.mean(contrasts))

pylab.plot(contrasts)
pylab.savefig("_".join(sys.argv[1:]) + "_" + str(np.mean(contrasts[-16:])) + ".png")