# Lithos

Lithos is a language designed to emit GPU opcode based on atomic mathematical functions which compose kernels, rather than compile them. It is designed to be able to tailor code to any model, and produce opcode in real time. The syntax is intentionally curt, avoids words and prefers symbols, and has an expansion goal to handle CPU operations.

In its current adolescent phase, the primary goal is to be able to write Lithos code that can be expressed, by whatever means, as the corresponding opcode. The initial design is a proof of concept that it can be used to bypass all intermediate layers and perform inference on Qwen 3.5 27B by direct opcode emission. Open questions remain as to whether missing pieces will be needed, but the sole focus now is to be able to write mathematical functions in Lithos, compose them into the needed kernels for the DeltaNet and Attention layers of the specified model, and ultimately produce a successful inference of a token.
