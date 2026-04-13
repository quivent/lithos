\\ scale_add.ls -- demonstrate composition (fusion)
\\ c[i] = a[i] * s[i] + b[i]   (scale-and-add, fused into one launch)

scale_add a b s c :
    each i
        c [ i ] = a [ i ] * s [ i ] + b [ i ]
