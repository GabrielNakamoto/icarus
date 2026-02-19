#include <stdio.h>
#include "icarus.h"

void print_2d_tensor(tensor *t) {
	for (int i=0; i<t->shape[0]; ++i) {
		for (int j=0; j<t->shape[1]; ++j) printf("%.2f ", t->data[j + i*t->shape[1]]);
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char **argv) {
	int shape_a[2] = { 3, 2 };
	int shape_b[2] = { 2, 3 };
	f32 data_a[6] = { 1, -10, 25, 0, 3, 4 };
	f32 data_b[6] = { 10, 5, 3, -2.7, 3.4, 0 };

	tensor *a = alloc_tensor(shape_a, 2);
	tensor *b = alloc_tensor(shape_b, 2);
	memcpy(a->data, data_a, 6 * sizeof(f32));
	memcpy(b->data, data_b, 6 * sizeof(f32));

	print_2d_tensor(a);
	print_2d_tensor(b);

	tensor *c = tensor_gemm(a, b);
	tensor *d = tensor_mul_scalar(c, 0.5);

	print_2d_tensor(c);
}
