\\ fused_scale_bias.ls -- y[i] = x[i] * scale[i] + bias[i]
\\ A single fused operation: scale, then add bias.
\\ In a traditional framework, this would be two separate kernel launches.
\\ In Lithos, it is one composition, one launch.

fused_scale_bias x scale bias y :
    each i
        y [ i ] = x [ i ] * scale [ i ] + bias [ i ]
