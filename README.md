# GPU 

## AUTHORS
Baptiste Bourdet \<baptiste.bourdet@epita.fr\> \
Hugo Moreau \<hugo.moreau@epita.fr\> \
Philippe Bernet \<philippe.bernet@epita.fr\> \

---

The objective of this project is to set up a **pipeline** to detect **objects** in multiple images looking for differences with a reference image. \

The pipeline computes the following steps: \
- greyscale the input image
- smooth the greyscaled image using a gaussian kernel
- compute the difference between the smoothed image and the smoothed image of the greyscaled reference image (the reference image is the same for all the images)
- compute an opening and then a closing with the output of the previous step
- binarize the output of the previous step
- compute the connected components and their bouding box of the binarized image

---

## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

It is possible to build everything with the following command: \

```bash
make -j
```

It is also possible to build the executable separately :

```bash
make main
make test
make bench
```

## Main

The program can be used as follows: \

```bash
./main [OPTIONS] -- REFENCE_IMAGE_PATH [IMAGE_PATH]*
```

Note that the dashs are mandatory. \

The program will then output on standard output a json with the following format \

```json
{
    "image_1": [
        [x1_1, y1_1, w1_1, h1_1],
        ...
        [x1_N, y1_N, w1_N, h1_N]
    ],
    ...
    "image_M": [
        [xM_1, yM_1, wM_1, hM_1],
        ...
        [xM_N, yM_N, wM_N, hM_N]
    ]
}
```

To know possible parameters of the program, you can use the following command: \

```bash
./main --help
```

---

## Test

Tests are done with Google Test and can be executed with the following command: \

```bash
./test
```

---

## Bench

Benchmarks are done with Google Benchmark and can be executed with the following command: \

```bash
./bench
```