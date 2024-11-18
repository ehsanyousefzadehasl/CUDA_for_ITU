# Assignment 1

1. Read the code for <CPU_code.c> and identify what the code does. <GPU_code.cu> does the same task on GPU. You need to pass different values as arguments to these CUDA programs after you compile them, and gather the result and compare them in a table similar to what you have on the lecture slides. How to compile:

```bash
g++ CPU_code.c -o CPU_code
nvcc GPU_code.cu -o GPU_code
```

How to run:

```bash
./CPU_code 1024
./GPU_code 1024
```

2. Do the same for the <MM.cu> code. How to compile:

```bash
nvcc MM.cu -o MM
```

How to run:

```bash
./MM 1024
```