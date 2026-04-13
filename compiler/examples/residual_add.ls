\\ residual_add.ls -- output[i] = projected[i] + residual[i]
\\ The simplest transformer pattern: adding a residual connection.

residual_add projected residual output :
    each i
        output [ i ] = projected [ i ] + residual [ i ]
