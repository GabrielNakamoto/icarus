#pragma once

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
	CREATE, RESHAPE, ADD, MUL, GEMM, RELU, IM2COL, CROSS_ENTROPY
} Op;


#define MAX_DIMS 4
#define SHAPE_2D(prefix, a, b) i32 prefix ## _shape[MAX_DIMS]; prefix ## _shape[0]=a; prefix ## _shape[1]=b; 
#define SHAPE_4D(prefix, a, b, c, d) i32 prefix ## _shape[MAX_DIMS]; prefix ## _shape[0]=a; prefix ## _shape[1]=b; prefix ## _shape[2]=c; prefix ## _shape[3] = d;

typedef struct {
	bool visited; void *grad;
	void *parent_l, *parent_r;
	Op op;

	i32 ndims, shape[MAX_DIMS], strides[MAX_DIMS];
	f32 *data;
} tensor;

typedef struct  {
	i32 inputs, outputs;
	tensor *weights, *bias;
} fc_layer;

typedef struct {
	i32 ksize, stride, in_channels, out_channels;
	tensor *kernel, *bias;
} conv2d_layer;

i32 get_size(tensor *t) {
	i32 size = 1;
	for (i32 i=0; i<t->ndims; ++i) size*=t->shape[i];
	return size;
}

typedef enum {
	TEMP,
	PARAM,
	SHALLOW
} alloc_type;

tensor *new_tensor(i32 ndims, i32 *shape, alloc_type type) {
	tensor *t = (tensor*)arena_alloc(sizeof(tensor));
	t->ndims = ndims;
	t->visited = false;
	t->op = CREATE;
	t->parent_l = NULL; t->parent_r = NULL;
	memcpy(t->shape, shape, ndims * sizeof(i32));
	i32 bytes = get_size(t) * sizeof(f32);
	f32 *buf;
	switch (type) {
		case TEMP:
			buf = (f32*)arena_alloc(bytes*2);
			t->data = buf;
			t->grad = buf + get_size(t);
			break;
		case PARAM:
			buf = (f32*)malloc(bytes*2);
			t->data = buf;
			t->grad = buf + get_size(t);
			break;
		default: break;
	}
	t->strides[ndims-1]=1;
	for (i32 i=ndims-2; i>=0; i--) t->strides[i]=t->shape[i+1]*t->strides[i+1];
	return t;
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
tensor *apply_biop2d(tensor *__restrict__ a, tensor *__restrict__ b, biop_fn fn, Op op) {
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
	c->op = op;

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
tensor *tensor_add(tensor *a, tensor *b) { return apply_biop2d(a, b, &op_add, ADD); }
tensor *tensor_mul(tensor *a, tensor *b) { return apply_biop2d(a, b, &op_mul, MUL); }

/*
* Cache friendly 2d general matrix multiply
*/
tensor *gemm2d(tensor *__restrict__ a, tensor *__restrict__ b) {
	if (a->shape[1] != b->shape[0]) return NULL;
	i32 M=a->shape[0], N=a->shape[1], P=b->shape[1];

	i32 shape[MAX_DIMS]; 
	shape[0] = M;
	shape[1] = P;

	tensor *c = new_tensor(2, shape, TEMP);
	c->op = GEMM;
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
tensor *relu(tensor *x, bool back) {
	i32 n = get_size(x);
	tensor *y = new_tensor(x->ndims, x->shape, TEMP);
	y->op = RELU;
	#pragma omp parallel for
	for (i32 i=0; i<n; ++i) y->data[i] = x->data[i] > 0 ? (back ? 1 : x->data[i]) : 0;
	return y;
}

/*
* Fused log_softmax and crossentropy loss forwards
*/
tensor *sparse_crossentropy(tensor *__restrict__ y, tensor *__restrict__ y_hat) {
	i32 N = y_hat->shape[0], C = y_hat->shape[1];
	SHAPE_2D(l, N, 1)
	tensor *l = new_tensor(2, l_shape, TEMP);
	l->op = CROSS_ENTROPY;
	l->parent_l = y_hat; l->parent_r = y;

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
 * Sparse cross entropy gradient computation
 */
void sparse_ce_backwards(tensor *__restrict__ node, tensor *__restrict__ grad, tensor *__restrict__ upstream) {
	tensor *y_hat = (tensor*)node->parent_l, *y = (tensor*)node->parent_r;
	i32 N=y_hat->shape[0], C=y_hat->shape[1];

	#pragma omp parallel for
	for (i32 i=0; i<N; ++i) {
		f32 *xr = y_hat->data + i*C;
		f32 *gr = grad->data + i*C;
		f32 up = upstream->data[i];

		f32 max = xr[0];
		for (i32 j=0; j<C; ++j) if(xr[j] > max) max = xr[j];
		f32 sum = 0.0f;
		for (i32 j=0; j<C; ++j) sum += expf(xr[j] - max);

		i32 label = (i32)y->data[i];
		for (i32 j=0; j<C; ++j)
			gr[j] = up * (expf(xr[j] - max) / sum - (j == label ? 1.0f : 0.0f));
	}
}

/*
* Shallow reshape, no data copy
*/
tensor *reshape(tensor *t, i32 ndims, i32 *shape) {
	tensor *r = new_tensor(ndims, shape, SHALLOW);
	r->op = RESHAPE;
	r->data = t->data;
	return r;
}

static inline i32 calculate_kernel_dim(i32 dimshape, i32 ksize, i32 kstride) { return floor((dimshape - ksize) / kstride)+1; }
/*
* Transforms 4d image tensor with form NHWC
* into 2d window collumns respecting kernel dimensions and stride
*/
tensor *im2col(tensor *__restrict__ im, i32 size, i32 strides) {
	i32 N=im->shape[0], H=im->shape[1], W=im->shape[2], C=im->shape[3];
	i32 oh = calculate_kernel_dim(H, size, strides), ow = calculate_kernel_dim(W, size, strides);
	SHAPE_2D(flat, N*oh*ow, size*size*C)
	tensor *cols = new_tensor(2, flat_shape, TEMP); 
	cols->op = IM2COL;

	#pragma omp parallel for collapse(3)
	for (i32 n=0; n<N; ++n) {
		for (i32 h=0; h<oh; ++h) {
			for (i32 w=0; w<ow; ++w) {
				i32 off_base = (n*oh*ow + h*ow + w) * cols->strides[0];
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
tensor *maxpool4d(tensor *x, i32 psize, i32 stride) {
	i32 N=x->shape[0], H=x->shape[1], W=x->shape[2], C=x->shape[3];
	i32 oh = calculate_kernel_dim(H, psize, stride), ow = calculate_kernel_dim(W, psize, stride);
	tensor *cols = im2col(x, psize, stride);
	SHAPE_4D(out, N, oh, ow, C)
	tensor *out = new_tensor(4, out_shape, TEMP);
	for (i32 i=0; i<cols->shape[0]; ++i) {
		for (i32 j=0; j<cols->shape[1]; ++j) {
		}
	}
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
	topo[*n++]=t;
	return;
}

/*
* Build computation graph and computes gradients
*/
tensor *backward(tensor *t) {
	tensor *topo[MAX_TOPO_NODES];
	i32 n;
	topodfs(t, &n, topo);

	for (i32 i=0; i<n-1; ++i) memset(((tensor*)topo[i]->grad)->data, 0.0f, get_size(topo[i]));
	for (i32 i=0; i<get_size(topo[n]); ++i) ((tensor*)topo[n]->grad)->data[i]=1.0f;
	for (i32 i=n-1; i>=0; i--) {
		tensor *node = topo[i];
		tensor *g = (tensor*)node->grad;
		switch (node->op) {
		}
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
	SHAPE_2D(w, outputs, inputs);
	SHAPE_2D(b, outputs, 1)
	layer->weights=new_tensor(2, w_shape, PARAM);
	layer->bias=new_tensor(2, b_shape, PARAM);

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
	SHAPE_2D(k, ksize*ksize*in_channels, out_channels)
	SHAPE_2D(b, out_channels, 1)
	layer->kernel = new_tensor(2, k_shape, PARAM);
	layer->bias = new_tensor(2, b_shape, PARAM);
	return layer;
}

tensor *conv2d_forward(conv2d_layer *layer, tensor *__restrict__ x) {
	tensor *cols = im2col(x, layer->ksize, layer->stride);
	return  tensor_add(gemm2d(cols, layer->kernel), layer->bias);
}
