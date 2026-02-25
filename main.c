#include <omp.h>
#include "icarus.h"
#include <stdio.h>

// https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

typedef struct {
	conv2d_layer *l1, *l2, *l4, *l5;
	batchnorm_layer *l3, *l6;
	fc_layer *l7;
	ADAM *adam;
} model;

static model net;

void model_init() {
	net.l1 = conv2d_init(1, 32, 5, 1);
	net.l2 = conv2d_init(32, 32, 5, 1);
	net.l3 = batchnorm_init(32);

	net.l4 = conv2d_init(32, 64, 3, 1);
	net.l5 = conv2d_init(64, 64, 3, 1);
	net.l3 = batchnorm_init(64);

	net.l7 = fc_init(576, 10);

	tensor *params[] = {
		net.l1->kernel, net.l1->bias,
		net.l2->kernel, net.l2->bias,
		net.l3->weights, net.l3->bias,

		net.l4->kernel, net.l4->bias,
		net.l5->kernel, net.l5->bias,
		net.l6->weights, net.l6->bias,

		net.l7->weights, net.l7->bias
	};

	net.adam = adam_init(14, params, 0.9, 0.99, 0.001);
}

/*
* Returns raw logits output
*/
tensor *model_forward(tensor *x) {
	x = relu(conv2d_forward(net.l1, x), false);
	x = conv2d_forward(net.l2, x);
	x = relu(batchnorm_forward(net.l3, x), false);
	x = maxpool4d(x, 2, 2);

	x = relu(conv2d_forward(net.l4, x), false);
	x = conv2d_forward(net.l5, x);
	x = relu(batchnorm_forward(net.l6, x), false);
	x = maxpool4d(x, 2, 2);

	return fc_forward(net.l7, x);
}

tensor *model_backward(tensor *y, tensor *y_hat) {
	tensor *loss = sparse_crossentropy(y, y_hat);
	backward(loss);
	adam_optimize(net.adam);
}

int main(int argc, char** argv) {
}
