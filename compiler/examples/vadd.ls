\\ vadd.ls -- vector addition: c[i] = a[i] + b[i]
\\ This is the simplest Lithos composition: one parallel loop, one operation.

vadd a b c :
    each i
        c [ i ] = a [ i ] + b [ i ]
