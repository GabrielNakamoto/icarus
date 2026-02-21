#include "icarus.h"
#include <stdio.h>


// https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
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

	tensor *params[14] = {
		net.l1->weights, net.l1->bias,
		net.l2->weights, net.l2->bias,
		net.l3->weights, net.l3->bias,
		net.l4->weights, net.l4->bias,
		net.l5->weights, net.l5->bias,
		net.l6->weights, net.l6->bias,
		net.l7->weights, net.l7->bias,
	};
	net.optim = ADAM_init(params, 14, 0.001, 0.9, 0.999);
}

tensor *model_forward(tensor *x) {
	x = tensor_relu(conv2d_forward(net.l1, x));
	x = conv2d_forward(net.l2, x);
	x = tensor_relu(batchnorm_forward(net.l3, x));

	x = tensor_relu(conv2d_forward(net.l4, x));
	x = conv2d_forward(net.l5, x);
	x = tensor_relu(batchnorm_forward(net.l6, x));

	i32 flattened[2] = { x->shape[0], 576 };
	x = tensor_reshape(x, flattened, 2);

	return linear_forward(net.l7, x);
}

f32 model_backward(tensor *y_hat, tensor *y) {
	tensor *loss = tensor_sparse_categorical_crossentropy_loss(y_hat, y);

	tensor_backward(loss);
	ADAM_step(net.optim);

	zero_grads(net.optim->params, net.optim->nparams);
}

// TODO: convert labels to one hot, maybe do this before serializing to raw floats
void load_dataset(tensor **t_images, tensor **t_labels, const char *images_filename, const char *labels_filename, i32 samples, i32 width, i32 height, i32 channels, i32 classes) {
	i32 ishape[4] = { samples, height, width, channels };
	i32 lshape[2] = { samples, 1 };

	*t_images = alloc_tensor(ishape, 4, 0, NEW, true);
	*t_labels = alloc_tensor(lshape, 2, 0, NEW, true);

	FILE *f_images = fopen(images_filename, "rb");
	FILE *f_labels = fopen(labels_filename, "rb");

	fread((*t_images)->data, 1, sizeof(f32) * samples * width * height * channels, f_images);
	fread((*t_labels)->data, 1, sizeof(f32) * samples * 1, f_labels);

	fclose(f_images);
	fclose(f_labels);;
}

void draw_mnist_digit(f32* data) {
    for (i32 y = 0; y < 28; y++) {
        for (i32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            i32 col = 232 + (i32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

int main(int argc, char **argv) {
	tensor *train_image_tensor, *train_label_tensor;
	load_dataset(&train_image_tensor, &train_label_tensor, "train_images.raw", "train_labels.raw", 60000, 28, 28, 1, 10);

	model_init();
	draw_mnist_digit(train_image_tensor->data);
}
