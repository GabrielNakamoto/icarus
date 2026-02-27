#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <omp.h>

#define MAX_TOPO_NODES 512
#define ARENA_SIZE (2UL*1024*1024*1024) // 2gb
#define ARENA_ALIGN 64
#define MAX_PARAMS 20

typedef float f32;
typedef int i32;
typedef long u64;
typedef char u8;

static u8 *arena_mem = NULL;
static size_t arena_offset = 0;

void arena_init() {
	arena_mem = (u8*)mmap(NULL, ARENA_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	if (arena_mem == MAP_FAILED) { perror("mmap"); exit(1); }
}

void arena_clear() { arena_offset = 0; }

void *arena_alloc(size_t size) {
	size_t padding = ARENA_ALIGN - (arena_offset % ARENA_ALIGN);
	if (arena_offset + padding + size > ARENA_SIZE) {
		printf("WARNING: Arena out of memory!\n");
		return NULL;
	}
	void *ptr = arena_mem + arena_offset + padding;
	arena_offset += padding + size;
	return ptr;
}

typedef enum {
	CREATE, RESHAPE, ADD, MUL, GEMM, RELU, IM2COL, CROSS_ENTROPY, MAX
} Op;


typedef enum {
	TEMP, PARAM, SHALLOW
} alloc_type;


#define MAX_DIMS 4
#define SHAPE_2D(prefix, a, b) i32 prefix ## _shape[MAX_DIMS]; prefix ## _shape[0]=a; prefix ## _shape[1]=b;
#define SHAPE_4D(prefix, a, b, c, d) i32 prefix ## _shape[MAX_DIMS]; prefix ## _shape[0]=a; prefix ## _shape[1]=b; prefix ## _shape[2]=c; prefix ## _shape[3]=d;

typedef struct tensor tensor;
typedef struct {
	tensor *dl, *dr;
} grads;
typedef grads (*backward_fn)(tensor *node, tensor *grad);
struct tensor {
	bool visited;
	tensor *grad;
	tensor *parent_l, *parent_r;
	backward_fn backward;
	i32 col2im_stride;

	alloc_type type;

	i32 ndims, shape[MAX_DIMS], strides[MAX_DIMS];
	f32 *data;
};

typedef struct {
	i32 inputs, outputs;
	tensor *weights, *bias;
} fc_layer;

typedef struct {
	i32 size;
	tensor *weights, *bias;
} batchnorm_layer;

typedef struct {
	i32 ksize, stride, in_channels, out_channels;
	tensor *kernel, *bias;
} conv2d_layer;

typedef struct {
	i32 nparams;
	tensor *m[MAX_PARAMS], *v[MAX_PARAMS], *params[MAX_PARAMS];
	f32 step_size, b1, b2;
	i32 step;
} ADAM;

i32 get_size(tensor *t) {
	i32 size = 1;
	for (i32 i=0; i<t->ndims; ++i) size*=t->shape[i];
	return size;
}


tensor *new_tensor(i32 ndims, i32 *shape, alloc_type type) {
	tensor *t = (tensor*)arena_alloc(sizeof(tensor));
	t->ndims = ndims;
	t->visited = false;
	t->parent_l = NULL; t->parent_r = NULL;
	t->grad = NULL;
	t->type = type;
	t->col2im_stride = 0;
	t->backward = NULL;
	memcpy(t->shape, shape, ndims * sizeof(i32));
	i32 bytes = get_size(t) * sizeof(f32);
	switch (type) {
		case TEMP: t->data = (f32*)arena_alloc(bytes); break;
		case PARAM: t->data = (f32*)malloc(bytes); break;
		default: break;
	}
	t->strides[ndims-1]=1;
	for (i32 i=ndims-2; i>=0; i--) t->strides[i]=t->shape[i+1]*t->strides[i+1];
	return t;
}

tensor *tensor2d(i32 r, i32 c, alloc_type t) {
	i32 shape[MAX_DIMS]; shape[0]=r; shape[1]=c;
	return new_tensor(2, shape, t);
}
tensor *tensor24(i32 n, i32 h, i32 w, i32 c, alloc_type t) {
	i32 shape[MAX_DIMS]; shape[0]=n; shape[1]=h; shape[2]=w; shape[3]=c;
	return new_tensor(4, shape, t);
}

/*
* 2d Tensor Cases:
* 1. Scalar, broadcast across other shape
* 2. Row broadcast (shape[0] == 1)
* 3. Col broadcast (shape[1] == 1)
* 4. Same shape
*/
typedef f32 (*biop_fn)(f32, f32);
static inline f32 op_add(f32 a, f32 b) { return a + b; }
static inline f32 op_mul(f32 a, f32 b) { return a * b; }

typedef enum { REDUCE_SUM, REDUCE_MAX } reduce_op;
grads backward_sum(tensor *node, tensor *grad) {
	tensor *x = node->parent_l, *dx = tensor2d(x->shape[0], x->shape[1], TEMP);
	i32 N=x->shape[0], M=x->shape[1];
	i32 axis = x->shape[0] == 1 ? 0 : 1;
	#pragma omp parallel for
	for (i32 i=0; i<N; ++i)
		for (i32 j=0; j<M; ++j)
			dx->data[i*M+j] = grad->data[axis == 0 ? j : i];
	return (grads){ .dl=dx, .dr=NULL };
}
grads backward_max(tensor *node, tensor *grad) {
	tensor *x = node->parent_l, *dx = tensor2d(x->shape[0], x->shape[1], TEMP);
	i32 N=x->shape[0], M=x->shape[1];
	i32 axis = x->shape[0] == 1 ? 0 : 1;
	i32 outer = axis == 0 ? M : N, inner = axis == 0 ? N : M;

	#pragma omp parallel for
	for (i32 i=0; i<outer; ++i) {
		i32 max_j = 0, max_v = -FLT_MAX;
		for (i32 j=0; j<inner; ++j) {
			f32 v = x->data[axis == 0 ? j*M+i : i*M+j];
			if (v > max_v) { max_v=v; max_j=j; }
		}
		dx->data[axis == 0 ? max_j*M+i : i*M+max_j]=grad->data[i];
	}
	return (grads){ .dl=dx, .dr=NULL };
}
tensor *reduce2d(tensor *x, i32 axis, reduce_op op) {
	i32 N=x->shape[0], M=x->shape[1];
	tensor *out = tensor2d(axis == 0 ? 1 : N, axis == 1 ? 1 : M, TEMP);
	i32 outer=axis == 0 ? M : N, inner = axis == 0 ? N : M;
	#pragma omp parallel for
	for (i32 i=0; i<outer; ++i) {
		out->data[i]=op==REDUCE_SUM ? 0 : -FLT_MAX;
		for (i32 j=0; j<inner; ++j) {
			f32 v = x->data[axis==0?j*M+i:i*M+j];
			if (op==REDUCE_SUM) out->data[i] += v;
			else if (v > out->data[i]) out->data[i]=v;
		}
	}
	out->parent_l = x;
	out->backward = op == REDUCE_SUM ? &backward_sum : &backward_max;
	return out;
}
#define sum_reduce2d(x, axis) reduce2d(x, axis, REDUCE_SUM)
#define max_reduce2d(x, axis) reduce2d(x, axis, REDUCE_MAX)

tensor *unbroadcast_grad2d(tensor *grad, tensor *target) {
	if (get_size(target) == 1) return sum_reduce2d(sum_reduce2d(grad, 0), 1);
	else if (target->shape[0] == 1) return sum_reduce2d(grad, 0);
	else if (target->shape[1] == 1) return sum_reduce2d(grad, 1);
	else return grad;
}

tensor *apply_biop2d(tensor *__restrict__ a, tensor *__restrict__ b, biop_fn fn, backward_fn backward) {
	if (a->ndims != b->ndims) return NULL;
	i32 a_size = get_size(a), b_size = get_size(b);
	bool a_scalar = a_size == 1, b_scalar = b_size == 1;
	bool a_row = a->shape[0] == 1, b_row = b->shape[0] == 1;
	bool a_col = a->shape[1] == 1, b_col = b->shape[1] == 1;
	bool same_shape = true;
	for (i32 i=0; i<a->ndims; ++i) if (a->shape[i] != b->shape[i]) same_shape=false;

	i32 cshape[MAX_DIMS];
	for (i32 i=0; i<a->ndims; ++i) cshape[i] = a->shape[i] > b->shape[i] ? a->shape[i] : b->shape[i];
	tensor *c = new_tensor(a->ndims, cshape, TEMP);
	c->parent_l = a; c->parent_r = b; c->backward = backward;

	if (a_scalar || b_scalar) {
		f32 s = a_scalar ? a->data[0] : b->data[0];
		i32 n = a_scalar ? b_size : a_size;
		f32 *big = a_scalar ? b->data : a->data;
		#pragma omp parallel for
		for (i32 i=0; i<n; ++i) c->data[i] = fn(s, big[i]);
	} else if (same_shape) {
		#pragma omp parallel for
		for (i32 i=0; i<a_size; ++i) c->data[i]= fn(a->data[i], b->data[i]);
	} else if (a_col || b_col) {
		f32 *col = a_col ? a->data : b->data;
		tensor *big = a_col ? b : a;
		i32 N=big->shape[0], M=big->shape[1];
		#pragma omp parallel for collapse(2)
		for (i32 i=0; i<N; ++i) 
			for (i32 j=0; j<M; ++j)
				c->data[i*M + j] = fn(col[i], big->data[i*M + j]);
	} else if (a_row || b_row) {
		f32 *row = a_row ? a->data : b->data;
		tensor *big = a_row ? b : a;
		i32 N=big->shape[0], M=big->shape[1];
		#pragma omp parallel for collapse(2)
		for (i32 i=0; i<N; ++i) 
			for (i32 j=0; j<M; ++j)
				c->data[i*M + j] = fn(row[j], big->data[i*M + j]);
	}
	return c;
}


tensor *tensor_mul(tensor *a, tensor *b);
grads backward_mul(tensor *node, tensor *grad) {
	tensor *lp = node->parent_l, *rp = node->parent_r;
	return (grads) {
		.dl = tensor_mul(unbroadcast_grad2d(grad, lp), lp),
		.dr = tensor_mul(unbroadcast_grad2d(grad, lp), rp)
	};
}
grads backward_add(tensor *node, tensor *grad) {
	return (grads) {
		.dl = unbroadcast_grad2d(grad, node->parent_l),
		.dr = unbroadcast_grad2d(grad, node->parent_r)
	};
}
tensor *tensor_add(tensor *a, tensor *b) { return apply_biop2d(a, b, &op_add, &backward_mul); }
tensor *tensor_mul(tensor *a, tensor *b) { return apply_biop2d(a, b, &op_mul, &backward_add); }

tensor *transpose2d(tensor *__restrict__ t) {
	tensor *__restrict__ T = tensor2d(t->shape[1], t->shape[0], TEMP);
	#pragma omp parallel for collapse(2)
	for (i32 i=0; i<t->shape[0]; ++i) 
			for (i32 j=0; j<t->shape[1]; ++j)
				T->data[j*t->shape[0] + i]=t->data[i*t->shape[1] + j];
	return T;
}

tensor *gemm2d(tensor *__restrict__ a, tensor *__restrict__ b);
grads backward_gemm(tensor *node, tensor *grad) {
	return (grads) {
		.dl = gemm2d(grad, transpose2d(node->parent_r)),
		.dr = gemm2d(transpose2d(node->parent_l), grad)
	};
}
/*
* Cache friendly 2d general matrix multiply
*/
tensor *gemm2d(tensor *__restrict__ a, tensor *__restrict__ b) {
	if (a->shape[1] != b->shape[0]) return NULL;

	i32 M=a->shape[0], N=a->shape[1], P=b->shape[1];
	tensor *c = tensor2d(M, P, TEMP);
	c->parent_l = a; c->parent_r = b; c->backward = &backward_gemm;
	memset(c->data, 0, M * P * sizeof(f32));

	#pragma omp parallel for
	for (i32 i=0; i<M; ++i) {
		for (i32 k=0; k<N; ++k) {
			float a_val = a->data[i*N + k];
			for (i32 j=0; j<P; ++j)
				c->data[i*P + j]+=a_val*b->data[k*P + j];
		}
	}
	return c;
}

/*
* Forward and backward rectified linear unit activation function
*/
tensor *_relu(tensor *x, bool back) {
	i32 n = get_size(x);
	tensor *y = new_tensor(x->ndims, x->shape, TEMP);
	y->parent_l = x;

	#pragma omp parallel for
	for (i32 i=0; i<n; ++i) y->data[i] = x->data[i] > 0 ? (back ? 1 : x->data[i]) : 0;
	return y;
}
grads backward_relu(tensor *node, tensor *grad) {
	return (grads) {
		.dl = tensor_mul(grad, _relu(node, true)),
		.dr = NULL
	};
}
tensor *relu(tensor *x) {
	tensor *y =  _relu(x, false);
	y->backward = &backward_relu;
	return y;
}

/*
* Fused log_softmax and crossentropy loss forwards
*/
grads backward_sparsece(tensor *node, tensor *grad) {
	tensor *y_hat = (tensor*)node->parent_l, *y = (tensor*)node->parent_r;
	i32 N=y_hat->shape[0], C=y_hat->shape[1];
	tensor *out = new_tensor(node->parent_l->ndims, node->parent_l->shape, TEMP);

	#pragma omp parallel for
	for (i32 i=0; i<N; ++i) {
		f32 *xr = y_hat->data + i*C;
		f32 *gr = out->data + i*C;
		f32 up = grad->data[i];

		f32 max = xr[0];
		for (i32 j=0; j<C; ++j) if(xr[j] > max) max = xr[j];
		f32 sum = 0.0f;
		for (i32 j=0; j<C; ++j) sum += expf(xr[j] - max);

		i32 label = (i32)y->data[i];
		for (i32 j=0; j<C; ++j)
			gr[j] = up * (expf(xr[j] - max) / sum - (j == label ? 1.0f : 0.0f));
	}
	return (grads) {
		.dl = out,
		.dr = NULL
	};
}
tensor *sparse_crossentropy(tensor *__restrict__ y, tensor *__restrict__ y_hat) {
	i32 N = y_hat->shape[0], C = y_hat->shape[1];
	tensor *l = tensor2d(N, 1, TEMP);
	l->parent_l = y_hat; l->parent_r = y; l->backward = &backward_sparsece;

	#pragma omp parallel for
	for (i32 i=0; i<N; ++i) {
		f32 *xr = y_hat->data + i*C;

		f32 max = xr[0];
		for (i32 j=0; j<C; ++j) if(xr[j] > max) max=xr[j];

		f32 sum = 0.0f;
		for (i32 j=0; j<C; ++j) sum += expf(xr[j] - max);
		f32 log_sum = logf(sum);

		i32 label = (i32)y->data[i];
		l->data[i] = -(xr[label] - max - log_sum);
	}
	return l;
}


/*
* Shallow reshape, no data copy
*/
tensor *reshape(tensor *t, i32 ndims, i32 *shape);
grads backward_reshape(tensor *node, tensor *grad) {
	return (grads) {
		.dl = reshape(grad, node->parent_l->ndims, node->parent_l->shape),
		.dr = NULL
	};
}
tensor *reshape(tensor *t, i32 ndims, i32 *shape) {
	tensor *r = new_tensor(ndims, shape, SHALLOW);
	r->parent_l = t; r->data = t->data; r->backward = &backward_reshape;
	return r;
}

static inline i32 calculate_kernel_dim(i32 dimshape, i32 ksize, i32 kstride) {
	return floor((dimshape - ksize) / kstride)+1;
}

/*
 * Backwards for col2im transform
 * Expands flattened windows into 4d window
 */
grads col2im(tensor *__restrict__ node, tensor *__restrict__ grad) {
	tensor *im = node->parent_l;
	i32 N=im->shape[0], H=im->shape[1], W=im->shape[2], C=im->shape[3];
	i32 os = sqrt(grad->shape[0]/N), ksize=sqrt(grad->shape[1]/C), stride=grad->col2im_stride;
	tensor *out = new_tensor(im->ndims, im->shape, TEMP);
	memset(out->data, 0, get_size(out) * sizeof(f32));

	// Iterate over output pos (n, h, w)
	#pragma omp parallel for
	for (i32 n=0; n<N; ++n) {
		i32 window_offset = n*os*os;
		for (i32 h=0; h<os; ++h) {
			for (i32 w=0; w<os; ++w) {
				i32 col_row_offset = h*os + w;
				i32 image_offset = n*H*W*C;
				// Iterate over kerenel position
				for (i32 i=0; i<ksize; ++i) {
					i32 row_offset = (h*stride + i)*W*C;
					for (i32 j=0; j<ksize; ++j) {
						i32 col_col_offset = (i*ksize+j)*C;
						i32 col_offset = (w*stride + j)*C;
						for (i32 c=0; c<C; ++c)
							out->data[image_offset + row_offset + col_offset + c] +=
								grad->data[(window_offset + col_row_offset)*grad->strides[0]  + col_col_offset + c];
					}
				}
			}
		}
	}
	return (grads) {
		.dl = out,
		.dr = NULL
	};
}
/*
* Transforms 4d image tensor with form NHWC
* into 2d window collumns respecting kernel dimensions and stride
*/
tensor *im2col(tensor *__restrict__ im, i32 size, i32 strides) {
	i32 N=im->shape[0], H=im->shape[1], W=im->shape[2], C=im->shape[3];
	i32 os = calculate_kernel_dim(H, size, strides);
	tensor *cols = tensor2d(N*os*os, size*size*C, TEMP); 
	cols->parent_l = im; cols->col2im_stride = strides; cols->backward = &col2im;

	#pragma omp parallel for collapse(3)
	for (i32 n=0; n<N; ++n) {
		for (i32 h=0; h<os; ++h) {
			for (i32 w=0; w<os; ++w) {
				i32 off_base = (n*os*os + h*os + w) * cols->strides[0];
				for (i32 i=0; i<size; ++i) {
					i32 row = (h*strides+i)*W*C;
					for (i32 k=0; k<size; ++k) {
						i32 col = (w*strides+k)*C;
						memcpy(
							&cols->data[off_base + i*size*C + k*C],
							&im->data[n*H*W*C + row + col],
							C*sizeof(f32)
						);
					}
				}
			}
		}
	}
	return cols;
}


/*
 * Max pooling layer on 4d image tensor (NHWC)
 */
grads backward_maxpool(tensor *node, tensor *grad) {
	tensor *cols = node->parent_l;
	tensor *im = cols->parent_l;
	i32 C = im->shape[3];
	i32 psize = sqrt(cols->shape[1] / C);

	tensor *dcols = tensor2d(cols->shape[0], cols->shape[1], TEMP);
	memset(dcols->data, 0, get_size(dcols) * sizeof(f32));

	#pragma omp parallel for
	for (i32 i=0; i<cols->shape[0]; ++i) {
		for (i32 c=0; c<C; ++c) {
			f32 max = -FLT_MAX;
			i32 max_j=0, max_k=0;
			for (i32 j=0; j<psize; ++j) {
				for (i32 k=0; k<psize; ++k) {
					f32 v = cols->data[i*psize*psize*C + j*psize*C + k*C + c];
					if (v > max) { max=v; max_j=j; max_k=k; }
				}
			}
			dcols->data[i*psize*psize*C + max_j*psize*C + max_k*C + c] = grad->data[i*C + c];
		}
	}

	dcols->parent_l = im;
	dcols->col2im_stride = node->col2im_stride;
	return col2im(dcols, dcols);
}
tensor *maxpool4d(tensor *x, i32 psize, i32 stride) {
	i32 N=x->shape[0], H=x->shape[1], W=x->shape[2], C=x->shape[3];
	i32 oh = calculate_kernel_dim(H, psize, stride), ow = calculate_kernel_dim(W, psize, stride);
	tensor *cols = im2col(x, psize, stride);
	tensor *out = tensor2d(N*oh*ow, C, TEMP);
	out->parent_l = cols; out->col2im_stride = stride; out->backward = &backward_maxpool;

	#pragma omp parallel for
	for (i32 i=0; i<cols->shape[0]; ++i) {
		for (i32 c=0; c<C; ++c) {
			f32 max = -FLT_MAX;
			for (i32 j=0; j<psize; ++j)
				for (i32 k=0; k<psize; ++k)
						if (cols->data[i*psize*psize*C + j*psize*C + k*C + c] > max)
								max = cols->data[i*psize*psize*C + j*psize*C + k*C + c];
			out->data[i*C + c]=max;
		}
	}
	SHAPE_4D(final, N, oh, ow, C)
	return reshape(out, 4, final_shape);
}

/*
 * Depth first search topo sort
 * - uses tensor visited member to reduce time complexity
 */
void topodfs(tensor *t, i32 *n, tensor *topo[]) {
	if (t->visited) return;
	t->visited = true;
	if (t->parent_l != NULL) topodfs((tensor*)t->parent_l, n, topo);
	if (t->parent_r != NULL) topodfs((tensor*)t->parent_r, n, topo);
	topo[(*n)++]=t;
	return;
}

void accum_grad(tensor *target, tensor *delta) {
	#pragma omp parallel for
	for (i32 i=0; i<get_size(target); ++i) target->data[i]+=delta->data[i];
}

/*
* Build computation graph and computes gradients
*/
tensor *backward(tensor *t) {
	tensor *topo[MAX_TOPO_NODES];
	i32 n = 0;
	topodfs(t, &n, topo);

	for (i32 i=0; i<n-1; ++i) {
		if (topo[i]->grad == NULL) topo[i]->grad = new_tensor(topo[i]->ndims, topo[i]->shape, topo[i]->type);
		memset(((tensor*)topo[i]->grad)->data, 0.0f, get_size(topo[i]) * sizeof(f32));
	}
	for (i32 i=0; i<get_size(topo[n-1]); ++i) ((tensor*)topo[n-1]->grad)->data[i]=1.0f;
	for (i32 i=n-1; i>=0; i--) {
		tensor *node = topo[i];
		if (!node->backward) continue;
		grads g = node->backward(node, node->grad);
		if (node->parent_l && g.dl) accum_grad(node->parent_l->grad, g.dl);
		if (node->parent_r && g.dr) accum_grad(node->parent_r->grad, g.dr);
	}
}

// Box-Mueller transform
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void he_init(f32 *data, i32 inputs, i32 outputs) {
	float std = sqrtf(2.0f / (float)inputs);
	for (i32 i=0; i<inputs*outputs; ++i) data[i]=randn() * std;
}

fc_layer *fc_init(i32 inputs, i32 outputs) {
	fc_layer *layer = (fc_layer*)malloc(sizeof(fc_layer));
	layer->inputs = inputs;
	layer->outputs = outputs;
	layer->weights=tensor2d(outputs, inputs, PARAM);
	layer->bias=tensor2d(outputs, 1, PARAM);

	he_init(layer->weights->data, inputs, outputs);
	return layer;
}

tensor *fc_forward(fc_layer *layer, tensor *__restrict__ x) {
	return tensor_add(gemm2d(layer->weights, x), layer->bias);
}

conv2d_layer *conv2d_init(i32 in_channels, i32 out_channels, i32 ksize, i32 stride) {
	conv2d_layer *layer = (conv2d_layer*)malloc(sizeof(conv2d_layer));
	layer->ksize = ksize; layer->stride = stride;
	layer->in_channels = in_channels; layer->out_channels = out_channels;
	layer->kernel = tensor2d(ksize*ksize*in_channels, out_channels, PARAM);
	layer->bias = tensor2d(out_channels, 1, PARAM);
	return layer;
}

tensor *conv2d_forward(conv2d_layer *layer, tensor *__restrict__ x) {
	tensor *cols = im2col(x, layer->ksize, layer->stride);
	return  tensor_add(gemm2d(cols, layer->kernel), layer->bias);
}

batchnorm_layer *batchnorm_init(i32 size) {
	batchnorm_layer *layer = (batchnorm_layer*)malloc(sizeof(batchnorm_layer));
	layer->size = size;
}

tensor *batchnorm_forward(batchnorm_layer *layer, tensor *__restrict__ x) {
}

ADAM *adam_init(i32 nparams, tensor **params, f32 b1, f32 b2, f32 step_size) {
	ADAM *adam = (ADAM*)malloc(sizeof(ADAM));
	adam->b1 = b1; adam->b2 = b2; adam->step_size = step_size;
	adam->nparams = nparams;
	adam->step = 0;
	for (i32 i=0; i<nparams; ++i) {
		adam->m[i] = new_tensor(params[i]->ndims, params[i]->shape, PARAM);
		adam->v[i] = new_tensor(params[i]->ndims, params[i]->shape, PARAM);
		adam->params[i]=params[i];
	}
	return adam;
}

void adam_optimize(ADAM *adam) {
	adam->step++;
	f32 mn = 1.0f - powf(adam->b1, adam->step), vn = 1.0f - powf(adam->b2, adam->step);
	#pragma omp parallel for
	for (i32 i=0; i<adam->nparams; ++i) {
		tensor *p = adam->params[i];
		f32 *gd = ((tensor*)p->grad)->data;
		i32 N = get_size(p);
		f32 *md = adam->m[i]->data, *vd = adam->v[i]->data;
		#pragma omp simd
		for (i32 j=0; j<N; ++j) {
			md[j]=adam->b1 * md[j] + (1 - adam->b1) * gd[j];
			vd[j]=adam->b2 * vd[j] + (1 - adam->b2) * gd[j] * gd[j];
			f32 mt = md[j] / mn, vt = vd[j] / vn;
			p->data[j] -= adam->step_size * mt / (sqrtf(vt) + 1e-8);
		}
	}
}
