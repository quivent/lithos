\ residual_add.li — output[i] = projected[i] + residual[i]
\ The simplest transformer pattern: adding a residual connection.

fn residual_add projected residual -> output
    each i
        output [ i ] = projected [ i ] + residual [ i ]
