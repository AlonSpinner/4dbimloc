# This is stupid, but we keep it here to remember that we can do it

import ifcopenshell
mystring = ifcopenshell.guid.expand('2pKX8dXSz3pRwIb_lNuNJ7')
print(mystring)

def binaryToDecimal(binary : str):
    dec = 0
    for i,b in enumerate(binary[::-1]):
        dec = dec + int(b)*2**i
    return dec

bitvec = ''.join([format(ord(char),'b') for char in mystring])
print(bitvec)
print(binaryToDecimal(bitvec))
