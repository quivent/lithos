\\ hello.ls -- minimal Lithos composition: vector add c[i] = a[i] + b[i]
\\ This is the simplest possible Lithos program: one composition, one operation.

vadd a b c :
    each i
        c [ i ] = a [ i ] + b [ i ]
