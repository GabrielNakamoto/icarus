#include <stdio.h>
#include "icarus.h"

int main(int argc, char **argv) {
	int shape_a[2] = { 3, 2 };
	int shape_b[2] = { 2, 3 };
	f32 data_a[6] = { 1, -10, 25, 0, 3, 4 };
	f32 data_b[6] = { 10, 5, 3, -2.7, 3.4, 0 };

	tensor *a = alloc_tensor(shape_a, 2);
	tensor *b = alloc_tensor(shape_b, 2);
	memcpy(a->data, data_a, 6 * sizeof(f32));
	memcpy(b->data, data_b, 6 * sizeof(f32));

	tensor *c = tensor_gemm(a, b);

	print_shape(c);
	for (int i=0; i<9; ++i) {
		printf("%f ", c->data[i]);
	}
}
