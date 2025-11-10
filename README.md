This program uses real Quantum random data straight from australian laboratory with quantum fluctations vaccuum

https://qrng.anu.edu.au/random-block-hex/ - you can check the information yourself

1) IT gets 2048 hex chars directly from https://qrng.anu.edu.au/wp-content/plugins/colours-plugin/get_block_hex.php

2) this whole string is being moved from beginning to end...last 5 hex chars are being scanned completely

3) if quantum random happens to generate the right combination at any point in those 2048 hex then its a jackpot


console example, first 13 hex(besides 1 prefix) of puzzle 73 and last 5 hex are completely scanned showing them as 00000


<img width="1453" height="639" alt="image" src="https://github.com/user-attachments/assets/0e81ab6c-60d7-44de-8301-4bcbfabf503f" />


Run (dont include puzzle number in the command, its just for info):

`main.exe 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56` - puzzle 21\
`main.exe 1000000 1ffffff 2f396b29b27324300d0c59b17c3abc1835bd3dbb` - puzzle 25\
`main.exe 10000000 1fffffff 5a416cc9148f4a377b672c8ae5d3287adaafadec` - puzzle 29\
`main.exe 20000000 3fffffff d39c4704664e1deb76c9331e637564c257d68a08` - puzzle 30\
`main.exe 100000000 1ffffffff 4e15e5189752d1eaf444dfd6bff399feb0443977` - puzzle 33\
`main.exe 200000000 3ffffffff f6d67d7983bf70450f295c9cb828daab265f1bfa` - puzzle 34\
`main.exe 1000000000000000000 1ffffffffffffffffff 105b7f253f0ebd7843adaebbd805c944bfb863e4` - puzzle 73

you can convert any bitcoin address to hash160 using this site: https://cointools.org/address-to-hash

just in case if you will edit this code further, you can compile it with the command:

`nvcc -o main main.cu`

also in secp256k1.cuh there are few parameters at top of the file:

`#define BIGINT_WORDS 8` - this one should stay as it is, cause 8 is required for 64 hex\
`#define WINDOW_SIZE 14` - this higher it is the more memory is required and the more loading time it will be before the start\
`#define NUM_BASE_POINTS 64` - its better not to change this, tough the less it is the less bits it can process for the correct hash160\
`#define BATCH_SIZE 64` - this one defined how deep the algorithm will search, if random defines X and Y coordinates, then this defines how deep it will go through Z coordinate\
`#define MOD_EXP 5` - this one affects the most heavy function, that is exponential calculation, the higher the faster it should be, but requires more memory, usually its 4,5,6

in case if this was useful to you, can you please donate BTC:

`bc1q8n38pk3urztlt4vceq0h089l9jdcw58l2c0e80`

so i will be more motivated to develop further this project

author: https://t.me/biernus
