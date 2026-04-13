\\ embed.ls — Token embedding lookup for LLM inference
\\ First operation in any transformer forward pass.
\\ Given a token ID, look up the corresponding row in the embedding table
\\ and write it to the output activation vector.
\\
\\ embed_table: [vocab_size, hidden_dim], FP16
\\ token_id: scalar u32 (the input token)
\\ output: [hidden_dim], FP32
\\
\\ Launch: gridDim.x = 1, blockDim.x = 256
\\ Each thread handles multiple elements via stride loop (hidden_dim=5120).

\\ ---- 1. token_embed: FP16 table, FP32 output --------------------------------
\\ output[i] = float(embed_table[token_id * hidden_dim + i])
\\ Stride loop covers hidden_dim with 256 threads.
token_embed embed_table output :
    param token_id u32
    param hidden_dim u32
    base = token_id * hidden_dim
    stride i hidden_dim
        offset = base + i
        cvt.f32.f16 val embed_table [ offset ]
        output [ i ] = val

\\ ---- 2. token_embed_f32: FP32 table, FP32 output ----------------------------
\\ output[i] = embed_table[token_id * hidden_dim + i]
\\ No conversion needed — straight load and store.
token_embed_f32 embed_table output :
    param token_id u32
    param hidden_dim u32
    base = token_id * hidden_dim
    stride i hidden_dim
        offset = base + i
        output [ i ] = embed_table [ offset ]
