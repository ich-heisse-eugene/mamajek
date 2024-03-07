#!/usr/local/bin/python3.11

"""
This programme interpolates the Eric Mamajek's table
"A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
to show the parameters like
Teff -- effective temperature, where [Teff] = K
log10 Teff -- logarithm of Teff, where [Teff] = K
log10 L -- logarithm of luminosity, where [L] = Lsun
Mbol -- bolometric magnitude
R -- radius, [R] = Rsun
Mv -- absolute magnitude
BV -- color index B-V in magnitudes
M -- mass, [M] = Msun
for a specified input value (or values separated by commas) of any of the
parameters specified above.

Additionally, this programe can evaluate apparent magnitude of a star or a binary system
located at the specified distance d, measured in pc or mas (with appropriate keyword), and account
for the interstellar reddening E(B-V).

To list the available keys, run this programme with a key -h or --help

The up-to-date original table is available on
http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

This programme uses the table updated on 2022.04.16.
Author: Eugene Semenko (eugene@narit.or.th)
Last update on 7 Mar 2024

Here is an example of using if we want to see all possible parameters of the individual components
with Teff = 13000 and 11500 K composing a binary system located at d = 372 pc with E(B-V) = 0.032 mag:
======================
./mamajek.py --given=Teff --val=13000,11500 --atdist=372 --Ebv=0.032 --binary

Programme must print:
Star #1: Teff = 13000.000	logTeff = 4.114	logL = 2.314	Mbol = -1.032	R = 2.803	Mv = -0.165	V = 7.8 (d = 372 pc, Av = 0.102)
BV = -0.114	M = 3.536	SpType: ['B7V', 'B8V']

Star #2: Teff = 11500.000	logTeff = 4.060	logL = 2.023	Mbol = -0.331	R = 2.692	Mv = 0.244	V = 8.2 (d = 372 pc, Av = 0.102)
BV = -0.100	M = 3.040	SpType: ['B7V', 'B8V']

Binary with components V1 = 7.79 mag and V2 = 8.20 mag appears as
a star with V = 7.37 mag at distance d = 372 pc, Av = 0.102 mag

Note: the effective temperature Teff is expressed in K, luminosity L, mass M, and radius R are in solar units. The rest of parameters are in magnitudes
========================

"""

import sys
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

sptype = ['O3V',  'O4V',  'O5V',  'O5.5V',  'O6V',  'O6.5V',  'O7V',  'O7.5V',  'O8V',  'O8.5V', \
          'O9V',  'O9.5V',  'B0V',  'B0.5V',  'B1V',  'B1.5V',  'B2V',  'B2.5V',  'B3V',  'B4V', \
          'B5V',  'B6V',  'B7V',  'B8V',  'B9V',  'B9.5V',  'A0V',  'A1V',  'A2V',  'A3V', \
          'A4V',  'A5V',  'A6V',  'A7V',  'A8V',  'A9V',  'F0V',  'F1V',  'F2V',  'F3V', \
          'F4V',  'F5V',  'F6V',  'F7V',  'F8V',  'F9V',  'F9.5V',  'G0V',  'G1V',  'G2V', \
          'G3V',  'G4V',  'G5V',  'G6V',  'G7V',  'G8V',  'G9V',  'K0V',  'K1V',  'K2V', \
          'K3V',  'K4V',  'K5V',  'K6V',  'K7V',  'K8V',  'K9V',  'M0V',  'M0.5V',  'M1V', \
          'M1.5V',  'M2V',  'M2.5V',  'M3V',  'M3.5V',  'M4V',  'M4.5V',  'M5V',  'M5.5V',  'M6V', \
          'M6.5V',  'M7V',  'M7.5V',  'M8V',  'M8.5V',  'M9V']

Teff = [44900,  42900,  41400,  40500,  39500,  38300,  37100,  36100,  35100,  34300, \
        33300,  31900,  31400,  29000,  26000,  24500,  20600,  18500,  17000,  16400, \
        15700,  14500,  14000,  12300,  10700,  10400,  9700,  9300,  8800,  8600, \
        8250,  8100,  7910,  7760,  7590,  7400,  7220,  7020,  6820,  6750, \
        6670,  6550,  6350,  6280,  6180,  6050,  5990,  5930,  5860,  5770, \
        5720,  5680,  5660,  5600,  5550,  5480,  5380,  5270,  5170,  5100, \
        4830,  4600,  4440,  4300,  4100,  3990,  3930,  3850,  3770,  3660, \
        3620,  3560,  3470,  3430,  3270,  3210,  3110,  3060,  2930,  2810, \
        2740,  2680,  2630,  2570,  2420,  2380]

logTeff = [4.652,  4.632,  4.617,  4.607,  4.597,  4.583,  4.569,  4.558,  4.545,  4.535, \
           4.522,  4.504,  4.497,  4.462,  4.415,  4.389,  4.314,  4.267,  4.230,  4.215, \
           4.196,  4.161,  4.146,  4.090,  4.029,  4.017,  3.987,  3.968,  3.944,  3.934, \
           3.917,  3.908,  3.898,  3.890,  3.880,  3.869,  3.859,  3.846,  3.834,  3.829, \
           3.824,  3.816,  3.803,  3.798,  3.791,  3.782,  3.777,  3.773,  3.768,  3.761, \
           3.757,  3.754,  3.753,  3.748,  3.744,  3.739,  3.731,  3.723,  3.713,  3.708, \
           3.684,  3.663,  3.647,  3.633,  3.613,  3.601,  3.594,  3.585,  3.576,  3.563, \
           3.559,  3.551,  3.540,  3.535,  3.515,  3.507,  3.493,  3.486,  3.467,  3.449, \
           3.438,  3.428,  3.420,  3.410,  3.384,  3.377]

BCv = [-4.010,  -3.890,  -3.760,  -3.670,  -3.570,  -3.490,  -3.410,  -3.330,  -3.240,  -3.180, \
       -3.110,  -3.010,  -2.990,  -2.830,  -2.580,  -2.440,  -2.030,  -1.770,  -1.540,  -1.490, \
       -1.340,  -1.130,  -1.050,  -0.730,  -0.420,  -0.360,  -0.210,  -0.140,  -0.070,  -0.040, \
       -0.020,   0.000,   0.005,   0.010,   0.020,   0.020,   0.010,   0.005,  -0.005,  -0.010, \
       -0.015,  -0.020,  -0.030,  -0.035,  -0.040,  -0.050,  -0.060,  -0.065,  -0.073,  -0.085, \
       -0.095,  -0.100,  -0.105,  -0.115,  -0.125,  -0.140,  -0.160,  -0.195,  -0.230,  -0.260, \
       -0.375,  -0.520,  -0.630,  -0.750,  -0.930,  -1.030,  -1.070,  -1.150,  -1.290,  -1.420, \
       -1.500,  -1.620,  -1.780,  -1.930,  -2.280,  -2.510,  -2.840,  -3.110,  -3.580,  -4.130, \
       -4.620,  -4.990,  -5.320,  -5.650,  -5.780,  -5.860]

logL = [5.82,  5.65,  5.54,  5.44,  5.36,  5.27,  5.18,  5.09,  4.99,  4.91, \
        4.82,  4.72,  4.65,  4.43,  4.13,  3.91,  3.43,  3.20,  2.99,  2.89, \
        2.77,  2.57,  2.48,  2.19,  1.86,  1.80,  1.58,  1.49,  1.38,  1.23, \
        1.13,  1.09,  1.05,  1.00,  0.96,  0.92,  0.86,  0.79,  0.71,  0.67, \
        0.62,  0.56,  0.43,  0.39,  0.29,  0.22,  0.18,  0.13,  0.08,  0.01, \
        -0.01,  -0.04,  -0.05,  -0.10,  -0.13,  -0.17,  -0.26,  -0.34,  -0.39,  -0.43, \
        -0.55,  -0.69,  -0.76,  -0.86,  -1.00,  -1.06,  -1.10,  -1.16,  -1.27,  -1.39, \
        -1.44,  -1.54,  -1.64,  -1.79,  -2.03,  -2.14,  -2.40,  -2.52,  -2.79,  -2.98, \
        -3.10,  -3.19,  -3.24,  -3.28,  -3.47,  -3.52]

Mbol = [-9.81,  -9.39,  -9.11,  -8.87,  -8.67,  -8.44,  -8.21,  -7.98,  -7.74,  -7.53, \
        -7.31,  -7.06,  -6.89,  -6.33,  -5.58,  -5.04,  -3.83,  -3.27,  -2.74,  -2.49, \
        -2.19,  -1.68,  -1.45,  -0.73,  0.08,  0.24,  0.78,  1.02,  1.28,  1.66, \
        1.92,  2.01,  2.13,  2.24,  2.34,  2.45,  2.58,  2.77,  2.97,  3.07, \
        3.19,  3.35,  3.66,  3.77,  4.01,  4.20,  4.29,  4.42,  4.55,  4.72, \
        4.78,  4.83,  4.88,  4.99,  5.08,  5.16,  5.39,  5.59,  5.72,  5.81, \
        6.13,  6.46,  6.65,  6.89,  7.23,  7.40,  7.49,  7.65,  7.91,  8.22, \
        8.35,  8.59,  8.82,  9.21,  9.82,  10.10,  10.74,  11.04,  11.72,  12.19, \
        12.48,  12.71,  12.84,  12.95,  13.42,  13.54]

Rsun = [13.43,  12.13,  11.45,  10.71,  10.27,  9.82,  9.42,  8.95,  8.47,  8.06, \
        7.72,  7.50,  7.16,  6.48,  5.71,  5.02,  4.06,  3.89,  3.61,  3.46, \
        3.36,  3.27,  2.94,  2.86,  2.49,  2.45,  2.19,  2.14,  2.12,  1.86, \
        1.79,  1.78,  1.77,  1.75,  1.75,  1.75,  1.73,  1.68,  1.62,  1.58, \
        1.53,  1.47,  1.36,  1.32,  1.22,  1.17,  1.14,  1.10,  1.06,  1.01, \
        1.00,  0.99,  0.98,  0.95,  0.93,  0.91,  0.85,  0.81,  0.80,  0.78, \
        0.76,  0.71,  0.70,  0.67,  0.63,  0.61,  0.61,  0.59,  0.54,  0.50, \
        0.48,  0.45,  0.42,  0.36,  0.30,  0.27,  0.22,  0.20,  0.16,  0.14, \
        0.13,  0.12,  0.12,  0.11,  0.10,  0.10]

Mv = [-5.80,  -5.50,  -5.35,  -5.20,  -5.10,  -4.95,  -4.80,  -4.65,  -4.50,  -4.35, \
      -4.20,  -4.05,  -3.90,  -3.50,  -3.00,  -2.60,  -1.80,  -1.50,  -1.20,  -1.00, \
      -0.85,  -0.55,  -0.40,  0.00,  0.50,  0.60,  0.99,  1.16,  1.35,  1.70, \
      1.94,  2.01,  2.12,  2.23,  2.32,  2.43,  2.57,  2.76,  2.97,  3.08, \
      3.20,  3.37,  3.69,  3.80,  4.05,  4.25,  4.35,  4.48,  4.62,  4.80, \
      4.87,  4.93,  4.98,  5.10,  5.20,  5.30,  5.55,  5.78,  5.95,  6.07, \
      6.50,  6.98,  7.28,  7.64,  8.16,  8.43,  8.56,  8.80,  9.20,  9.64, \
      9.85,  10.21,  10.61,  11.15,  12.10,  12.61,  13.58,  14.15,  15.30,  16.32, \
      17.10,  17.70,  18.16,  18.60,  19.20,  19.40]

BV = [-0.330,  -0.326,  -0.323,  -0.322,  -0.321,  -0.319,  -0.318,  -0.317,  -0.315,  -0.314, \
      -0.312,  -0.307,  -0.301,  -0.289,  -0.278,  -0.252,  -0.215,  -0.198,  -0.178,  -0.165, \
      -0.156,  -0.140,  -0.128,  -0.109,  -0.070,  -0.050,  0.000,  0.035,  0.070,  0.100, \
      0.140,  0.160,  0.185,  0.210,  0.250,  0.270,  0.295,  0.330,  0.370,  0.390, \
      0.410,  0.440,  0.486,  0.500,  0.530,  0.560,  0.580,  0.595,  0.622,  0.650, \
      0.660,  0.670,  0.680,  0.700,  0.710,  0.730,  0.775,  0.816,  0.857,  0.884, \
      0.990,  1.090,  1.150,  1.240,  1.340,  1.363,  1.400,  1.420,  1.445,  1.485, \
      1.495,  1.505,  1.522,  1.530,  1.600,  1.650,  1.690,  1.830,  1.940,  2.010, \
      2.070,  2.120,  2.140,  2.150,  2.160,  2.170]

Msun = [59.00,  48.00,  43.00,  38.00,  35.00,  31.00,  28.00,  26.00,  23.60,  21.90, \
        20.20,  18.70,  17.70,  14.80,  11.80,  9.90,  7.30,  6.10,  5.40,  5.10, \
        4.70,  4.30,  3.92,  3.38,  2.75,  2.68,  2.18,  2.05,  1.98,  1.86, \
        1.93,  1.88,  1.83,  1.77,  1.81,  1.75,  1.61,  1.50,  1.46,  1.44, \
        1.38,  1.33,  1.25,  1.21,  1.18,  1.13,  1.08,  1.06,  1.03,  1.00, \
        0.99,  0.98,  0.98,  0.97,  0.95,  0.94,  0.90,  0.88,  0.86,  0.82, \
        0.78,  0.73,  0.70,  0.69,  0.64,  0.62,  0.59,  0.57,  0.54,  0.50, \
        0.47,  0.44,  0.40,  0.37,  0.27,  0.23,  0.18,  0.16,  0.12,  0.10, \
        0.09,  0.09,  0.09,  0.09,  0.08,  0.08]

def interpolate_in(par_prim, par_seg, valor):
    par_prim = np.asarray(par_prim)
    par_seg = np.asarray(par_seg)
    margin = 5
    idx = (np.abs(valor - par_prim)).argmin()
    if idx >= margin and idx <= len(par_prim) - margin:
        idx_peq = np.linspace(idx-margin, idx+margin, 2*margin+1, dtype=int)
    elif idx < margin:
        idx_peq = np.linspace(0, idx+margin, idx+margin, dtype=int)
    elif idx > len(par_prim) - margin:
        idx_peq = np.linspace(idx-margin, len(par_prim)-1, len(par_prim)-idx+margin, dtype=int)
    prim = par_prim[idx_peq]; seg = par_seg[idx_peq]
    srt = prim.argsort()
    spl = CubicSpline(prim[srt], seg[srt])
    return spl(valor), idx


if __name__ == "__main__":
    convers = {'Teff': Teff, 'logTeff': logTeff, 'logL': logL, 'Mbol': Mbol, 'R': Rsun, 'Mv': Mv, 'BV': BV, 'M': Msun}

    parser = argparse.ArgumentParser()
    parser.add_argument("--given", help="Input parameter, one out of [Teff, logTeff, logL, Mbol, R, Mv, BV, M, V]", type=str, default=None)
    parser.add_argument("--val", help="Input value or values separated by commas", type=str, default=None)
    parser.add_argument("--atdist", help="Distance in pc or parallax in mas. The latter requires key --mas", type=float, default=None)
    parser.add_argument("--mas", help="Given distance is parallax in mas", action="store_true")
    parser.add_argument("--Ebv", help="Interstellar reddening E(B-V) in magnitudes", type=float, default=0)
    parser.add_argument("--binary", help="Apparent magnitude is calculated for a binary located at specified distance. It requires at least two input values and only two first stars are used", \
                        action="store_true")
    args = parser.parse_args()

    if args.given == None:
        print("El programa fue loco. No parameters specified. Run this programme with -h")
        sys.exit(1)

    if args.val == None:
        print("Hmmm. No values specified. Run this programme with -h and read help. ¡Hasta luego!")

    if args.val.find(',') != -1:
        values = np.asarray(args.val.split(','), dtype=float)
        nval = len(values)
    else:
        values = np.asarray([args.val], dtype=float)
        nval = 1

    if args.atdist != None and args.mas == True:
        d = 1000. / args.atdist
    else:
        d = args.atdist
    Av = 3.2 * args.Ebv

    if args.binary:
        mbin = np.zeros(2)
        Lbin = np.zeros(2)

    if args.given == 'V':
        print("The apparent magnitude is given as an input parameter. Computing absolute magnitude(s)... ")
        print(f"Distance d is {d} pc")
        args.given = 'Mv'
        values = values + 5 - 5*np.log10(d)

    for i in range(nval):
        print(f"Star #{i+1}: {args.given} = {values[i]:.3f}\t", end='')
        for key, val in convers.items():
            if key != args.given:
                newval, idx = interpolate_in(convers[args.given], val, values[i])
                print(f"{key} = {newval:.3f}\t", end='')
            if args.atdist != None and (args.given == 'Mv' or key == 'Mv'):
                if args.given == 'Mv':
                    Mv = values[i]
                elif key == 'Mv':
                    Mv = newval
                mv = Mv - 5. + 5. * np.log10(d) + Av
                print(f"V = {mv:.1f} (d = {d:.0f} pc, Av = {Av:.3f})\t")
                if args.binary and nval >= 2 and i <= 1:
                    mbin[i] = mv
                    Lbin[i], _ = interpolate_in(convers[args.given], convers['logL'], values[i])
        print(f"SpType: {sptype[idx-1:idx+1]}\n")

    if args.binary and nval >= 2:
        Vbin = mbin[1] - 2.5 * np.log10((Lbin[0] + Lbin[1]) / Lbin[1])
        print(f"Binary with components V1 = {mbin[0]:.2f} mag and V2 = {mbin[1]:.2f} mag appears as")
        print(f"a star with V = {Vbin:.2f} mag at distance d = {d:.0f} pc, Av = {Av:.3f} mag\n")

    print("Note: the effective temperature Teff is expressed in K, luminosity L, mass M, and radius R are in solar units. The rest of parameters are in magnitudes")
