"""
Color corrections for photometry
The nature of encoding transformations should be clear from these examples
   K_MKO = K_2MASS -0.003 (+/-0.007) - 0.026 (+/-0.011) * (J-K)_2MASS
   K_MKO = K_2MASS -0.006 (+/-0.004) - 0.071 (+/-0.020) * (H-K)_2MASS
           Leggett 2008,  http://arxiv.org/pdf/astro-ph/0609461v1.pdf
   K(prime) = K_MKO + 0.22 (+/- 0.003) * (H-K)_2MASS
           (Wainscoat and Cowie 1992AJ.103.332W)
If multiple options, make a list of lists; code will choose options
with smallest uncertainty on an object-by-object basis
The filter names are case-sensitive
The catalog column names are not
"""

colorTerms = {
    "u": ["u"],
    "g": ["g"],
    "r": ["r"],
    "i": ["i"],
    "z": ["z"],
    "J": ["j"],
    "H": ["h"],
    "Kshort": ["k"],
    "K(short)": ["k"],
    "Ks": ["k"],
    "K": [
        ["k", (-0.003, 0.007), (-0.026, 0.011, "j-k")],
        ["k", (-0.006, 0.004), (-0.071, 0.020, "h-k")],
    ],
    "Kprime": ["k", (-0.006, 0.007), (0.149, 0.023, "h-k")],
    "K(prime)": ["k", (-0.006, 0.007), (0.149, 0.023, "h-k")],
}
