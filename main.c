#include "icarus.h"


typedef struct {
	layer_conv2d *l1, *l2;
	layer_batchnorm *l3;

	layer_conv2d *l4, *l5;
	layer_batchnorm *l6;

	layer_linear *l7;

	optim_ADAM *optim;
} model;

static model net;

void model_init() {
	net.l1 = conv2d_init(1, 32, 1, 5, 0);
	net.l2 = conv2d_init(32, 32, 1, 5, 0);
	net.l3 = batchnorm_init(32);

	net.l4 = conv2d_init(32, 64, 1, 3, 0);
	net.l5 = conv2d_init(64, 64, 1, 3, 0);
	net.l6 = batchnorm_init(64);

	net.l7 = linear_init(576, 10);
}
void model_forward(tensor *x) {
	x = conv2d_forward(net.l1, x);
	x = conv2d_forward(net.l2, x);
	x = batchnorm_forward(net.l3, x);

	x = conv2d_forward(net.l4, x);
	x = conv2d_forward(net.l5, x);
	x = batchnorm_forward(net.l6, x);

	i32 flattened[2] = { x->shape[0], 576 };
	x = tensor_reshape(x, flattened, 2);

	x = linear_forward(net.l7, x);
}

int main(int argc, char **argv) {
}
