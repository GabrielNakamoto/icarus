#include "icarus.h"

int main(int argc, char **argv) {
	int shape_a[2] = { 3, 2 };
	int shape_b[2] = { 2, 3 };
	f32 data_a[6] = { 1, -10, 25, 0, 3, 4 };
	f32 data_b[6] = { 10, 5, 3, 22.7, 3.4, 0 };

	tensor *a = alloc_tensor(shape_a, 2, 0, NEW);
	tensor *b = alloc_tensor(shape_b, 2, 0, NEW);
	memcpy(a->data, data_a, 6 * sizeof(f32));
	memcpy(b->data, data_b, 6 * sizeof(f32));

	print_2d_tensor(a);
	print_2d_tensor(b);

	// tensor *c = tensor_add(b, tensor_mul_scalar(tensor_relu(a), 0.75));
	tensor *c = tensor_gemm(a, b);
	print_2d_tensor(c);
	tensor_backward(c);

	printf("A grad\n");
	print_2d_tensor(a->grad);
	printf("B grad\n");
	print_2d_tensor(b->grad);
}
