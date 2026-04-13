\ vadd.li — vector addition: c[i] = a[i] + b[i]
\ This is the simplest Lithos function: one parallel loop, one operation.

fn vadd a b -> c
    each i
        c [ i ] = a [ i ] + b [ i ]
