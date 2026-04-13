\ fused_scale_bias.li — y[i] = x[i] * scale[i] + bias[i]
\ A single fused operation: scale, then add bias.
\ In a traditional framework, this would be two separate kernel launches.
\ In Lithos, it is one function, one launch.

fn fused_scale_bias x scale bias -> y
    each i
        y [ i ] = x [ i ] * scale [ i ] + bias [ i ]
