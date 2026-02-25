#include <omp.h>
#include "icarus.h"
#include <stdio.h>

// https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

typedef struct {
	conv2d_layer l1, l2, l4, l5;
	batchnorm_layer l3, l6;
	fc_layer l7;
} model;

int main(int argc, char** argv) {
}
