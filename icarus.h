#pragma once 

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define MAX_TOPO_NODES 1024
#define ARENA_SIZE 16*1024*1024
#define ARENA_ALIGN 64

typedef float f32;
typedef int i32;
typedef long u64;
typedef char u8;

static u8 arena_mem[ARENA_SIZE];
static size_t arena_offset = 0;

void arena_clear() {
	printf("Arena bytes used before clear: %ld\n", arena_offset);
	arena_offset = 0;
}
void *arena_alloc(size_t size) {
	size_t padding = ARENA_ALIGN - (arena_offset % ARENA_ALIGN);
	// printf("Allocating %ld bytes on arena with padding: %ld...\n", size, padding);
	if (arena_offset + padding + size > ARENA_SIZE) return NULL;
	void *ptr = arena_mem + arena_offset + padding;
	arena_offset += padding + size;
	return ptr;
}

typedef enum {
	NEW, RESHAPE,
	POW, EXP, LOG,
	MUL, ADD,
	SUM, MAX,
	GEMM, RELU
} tensor_op;

const char* const op_names[] = {
    "NEW", "RESHAPE", "POW", "EXP", "LOG", "MUL",
		"ADD", "SUM", "MAX", "GEMM", "RELU",
};

typedef union {
	void *t;
	f32 s;
} tensor_parents_value;

typedef enum {
	TENSOR, SCALAR, NONE
} tensor_parent_type;

const char* const parent_type_names[] = {
	"TENSOR", "SCALAR", "NONE"
};

typedef struct {
	tensor_parents_value value;
	tensor_parent_type type;
} tensor_parent;

typedef struct {
	// Autograd fields
	tensor_parent parent_r, parent_l;
	tensor_op parent_op;
	f32 grad_arg;
	bool grad_keptdims, is_param;
	void *grad; // tensor

	// Metadata
	i32 ndims;
	i32 *shape, *strides;

	f32 *data;
} tensor;


tensor *alloc_tensor(i32 *shape, i32 ndims, f32 init, tensor_op op, bool is_param);
f32 *tensor_getitem(tensor *t, i32 *strides, i32 *shape);

// Reshape Ops
tensor *tensor_reshape(tensor *t, i32 *newshape, i32 newndims);

// Unary Ops
tensor *tensor_pow(tensor *t, f32 n);
tensor *tensor_exp(tensor *t);
tensor *tensor_log(tensor *t);
tensor *tensor_sqrt(tensor *t);

// Binary Ops
tensor *tensor_mul(tensor *a, tensor *b);
tensor *tensor_mul_scalar(tensor *a, f32 b);
tensor *tensor_div(tensor *a, tensor *b);
tensor *tensor_add(tensor *a, tensor *b);
tensor *tensor_sub(tensor *a, tensor *b);

// Reduce Ops
tensor *tensor_sum(tensor *t, i32 axis, bool keepdims);
tensor *tensor_mean(tensor *t, i32 axis, bool keepdims);
tensor *tensor_max(tensor *t, i32 axis, bool keepdims);

// Composed Ops
tensor *tensor_gemm(tensor *a, tensor *b);
tensor *tensor_softmax(tensor *t);
tensor *tensor_relu(tensor *t);

// CNN Ops
tensor *tensor_im2col(tensor *im, int kh, int kw, int sh, int sw, int ph, int pw);

// Helper funcs
i32 get_size(i32 *shape, i32 ndims) {
	i32 size = 1;
	for (i32 i=0; i<ndims; ++i) size *= shape[i];
	return size;
}

void copy_data(tensor *dst, tensor *src) { memcpy(dst->data, src->data, get_size(src->shape, src->ndims) * sizeof(f32)); }

void calculate_strides(i32 *shape, i32 ndims, i32 **strides) {
	(*strides)[ndims-1]=1;
	for (i32 i=ndims-2; i>=0; i--) (*strides)[i]=shape[i+1]*(*strides)[i+1];
}

i32 *broadcast_strides(tensor *t) {
	i32 *bstrides = (i32*)arena_alloc(t->ndims * sizeof(i32));
	for (i32 i=0; i<t->ndims; ++i) bstrides[i] = t->shape[i] == 1 ? 0 : t->strides[i];
	return bstrides;
}

i32 *broadcast_shape(i32 *a_sh, i32 *b_sh, i32 ndims) {
	i32 *bshape = (i32*)arena_alloc(ndims * sizeof(i32));
	for (i32 i=0; i<ndims; ++i) bshape[i] = a_sh[i] == 1 ? b_sh[i] : a_sh[i];
	return bshape;
}

i32 tensor_getidx(i32 *strides, i32 ndims, i32 *indices) {
	i32 idx = 0;
	for (i32 i=0; i<ndims; ++i) idx += strides[i] * indices[i];
	return idx;
}

f32 *tensor_getitem(tensor *t, i32 *strides, i32 *indices) { return t->data + tensor_getidx(strides, t->ndims, indices); }

i32 inc_shapeindex(i32 *indices, i32 *shape, i32 ndims) {
	indices[ndims-1]++;
	for (i32 i=ndims-1; i>=0; i--) {
		if (indices[i] != shape[i]) break;
		indices[i]=0;
		if (i == 0) return -1;
		indices[i-1]++;
	}
	return 1;
}

tensor *tensor_reshape(tensor *t, i32 *newshape, i32 ndims) {
	if (get_size(newshape, ndims) != get_size(t->shape, t->ndims)) return NULL;

	tensor *nt = alloc_tensor(newshape, ndims, 0, RESHAPE, false);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	memcpy(nt->data, t->data, get_size(t->shape, t->ndims) * sizeof(f32));
	return nt;
}

tensor *tensor_apply_unop(tensor *t, f32 (*func)(f32, f32), f32 arg, tensor_op op) {
	tensor *nt = alloc_tensor(t->shape, t->ndims, 0, op, false);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	nt->parent_r.type = NONE; nt->grad_arg = arg;

	i32 *iter = (i32*)arena_alloc(t->ndims * sizeof(i32));
	memset(iter, 0, t->ndims * sizeof(i32));
	do {
		f32 *b = tensor_getitem(nt, nt->strides, iter);
		*b = func(*tensor_getitem(t, t->strides, iter), arg);
	} while(inc_shapeindex(iter, t->shape, t->ndims) != -1);
	return nt;
}

f32 _exp(f32 x, f32 _) { return (f32) expf(x); }
f32 _log(f32 x, f32 _) { return (f32) logf(x); }
f32 _pow(f32 x, f32 n) { return (f32) powf(x, n); }
f32 _relu(f32 x, f32 _) { return x > 0 ? x : 0; }
f32 _relu_back(f32 x, f32 _) { return x > 0 ? 1.0f : 0; }
tensor *tensor_exp(tensor *t) { return tensor_apply_unop(t, &_exp, -1, EXP); }
tensor *tensor_log(tensor *t) { return tensor_apply_unop(t, &_log, -1, LOG); }
tensor *tensor_pow(tensor *t, f32 n) { return tensor_apply_unop(t, &_pow, n, POW); }
tensor *tensor_sqrt(tensor *t) { return tensor_pow(t, 0.5); }
tensor *tensor_relu(tensor *t) { return tensor_apply_unop(t, &_relu, -1, RELU); }
tensor *_tensor_reluback(tensor *t) { return tensor_apply_unop(t, &_relu_back, -1, RELU); }

tensor *tensor_apply_biop(tensor *a, tensor *b, f32 (*func)(f32, f32), tensor_op op) {
	if (a->ndims != b->ndims) return NULL;
	i32 *cshape = broadcast_shape(a->shape, b->shape, a->ndims);
	tensor *c = alloc_tensor(cshape, a->ndims, 0, op, false);

	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = TENSOR; c->parent_r.value.t = b;

	i32 *a_bstrides = broadcast_strides(a); i32 *b_bstrides = broadcast_strides(b);
	i32 *iter = (i32*)arena_alloc(a->ndims * sizeof(i32));
	memset(iter, 0, a->ndims * sizeof(i32));
	do {
		f32 *cv = tensor_getitem(c, c->strides, iter);
		*cv = func(*tensor_getitem(a, a_bstrides, iter), *tensor_getitem(b, b_bstrides, iter));
	} while (inc_shapeindex(iter, c->shape, c->ndims) != -1);

	return c;
}

tensor *tensor_apply_biop_scalar(tensor *a, f32 b, f32 (*func)(f32, f32), tensor_op op) {
	tensor *c = alloc_tensor(a->shape, a->ndims, 0, op, false);

	i32 *a_bstrides = broadcast_strides(a);
	i32 *iter = (i32*)arena_alloc(a->ndims * sizeof(i32));
	memset(iter, 0, a->ndims * sizeof(i32));

	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = SCALAR; c->parent_r.value.s = b;

	do {
		f32 *cv = tensor_getitem(c, c->strides, iter);
		*cv = func(*tensor_getitem(a, a_bstrides, iter), b);
	} while (inc_shapeindex(iter, c->shape, c->ndims) != -1);

	return c;
}

f32 __mul(f32 a, f32 b) { return a * b; }
f32 __add(f32 a, f32 b) { return a + b; }
f32 __eq(f32 a, f32 b) { return a == b ? 1 : 0; }
tensor *tensor_mul(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__mul, MUL); }
tensor *tensor_div(tensor *a, tensor *b) { return tensor_mul(a, tensor_pow(b, -1)); }
tensor *tensor_add(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__add, ADD); }
tensor *tensor_sub(tensor *a, tensor *b) { return tensor_add(a, tensor_mul_scalar(b, -1)); }
tensor *tensor_eq(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__eq, NEW); }
tensor *tensor_mul_scalar(tensor *a, f32 b) { return tensor_apply_biop_scalar(a, b, &__mul, MUL); }
tensor *tensor_add_scalar(tensor *a, f32 b) { return tensor_apply_biop_scalar(a, b, &__add, ADD); }

tensor *tensor_apply_reduceop(tensor *t, i32 axis, bool keepdims, void (*func)(f32*, f32), f32 init, tensor_op op) {
	i32 *nshape;
	if (! keepdims) {
		nshape = (i32*)arena_alloc((t->ndims-1) * sizeof(i32));
		for (i32 i=0; i<axis; ++i) nshape[i]=t->shape[i];
		for (i32 j=axis; j<t->ndims-1; ++j) nshape[j]=t->shape[j+1];
	} else {
		nshape = (i32*)arena_alloc(t->ndims * sizeof(i32));
		for (i32 i=0; i<t->ndims; ++i) nshape[i] = i == axis ? 1 : t->shape[i];
	}
	tensor *r = alloc_tensor(nshape, keepdims ? t->ndims : t->ndims-1, 0, op, false);

	r->parent_l.type = TENSOR; r->parent_l.value.t = t;
	r->parent_r.type = NONE;
	r->grad_keptdims = keepdims; r->grad_arg = axis;

	i32 *riter = (i32*)arena_alloc(r->ndims * sizeof(i32)), *iter = (i32*)arena_alloc(t->ndims * sizeof(i32));
	memset(riter, 0, r->ndims * sizeof(i32));

	do {
		if (!keepdims) {
			for (i32 i=0; i<r->ndims; ++i) iter[i + (i >= axis)]=riter[i];
		} else memcpy(iter, riter, t->ndims * sizeof(i32));
		f32 *a = tensor_getitem(r, r->strides, riter);
		*a = init;
		for (i32 i=0; i<t->shape[axis]; ++i) {
			iter[axis]=i;
			func(a, *tensor_getitem(t, t->strides, iter));
		}
	} while (inc_shapeindex(riter, r->shape, r->ndims) != -1);
	return r;
}

void __sum(f32 *a, f32 b) { *a += b; }
void __max(f32 *a, f32 b) { *a = *a > b ? *a : b; }
tensor *tensor_sum(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__sum, 0, SUM); }
tensor *tensor_max(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__max, -INFINITY, MAX); }
tensor *tensor_mean(tensor *t, i32 axis, bool keepdims) { return tensor_mul_scalar(tensor_sum(t, axis, keepdims), 1.0 / (f32)t->shape[axis]); }

tensor *tensor_gemm(tensor *a, tensor *b) {
	if (a->shape[1] != b->shape[0]) return NULL;
	i32 ash[3] = { a->shape[0], a->shape[1], 1 };
	i32 bsh[3] = { 1, b->shape[0], b->shape[1] };

	a = tensor_reshape(a, ash, 3);
	b = tensor_reshape(b, bsh, 3);
	tensor *c = tensor_mul(a, b);

	return tensor_sum(c, 1, false);
}

tensor *tensor_logsoftmax(tensor *t) {
	tensor *cache = tensor_sub(t, tensor_max(t, 1, true));
	tensor *e = tensor_exp(cache);
	return tensor_sub(cache, tensor_log(tensor_sum(e, 1, true)));
}

tensor *tensor_sparse_categorical_crossentropy_loss(tensor *y_hat, tensor *y) {
	tensor *props = tensor_logsoftmax(y_hat);
	tensor *classes = tensor_sum(tensor_mul(y, props), 1, true);
	return tensor_mul_scalar(tensor_mean(classes, 1, true), -1.0);
}

void topo_dfs(tensor *t, tensor **topo, tensor **seen, i32 *n, i32 *ns) {
	for (i32 i=0; i<*ns; ++i) if (seen[i] == t) return;
	seen[(*ns)++]=t;
	if (t->parent_l.type == TENSOR) topo_dfs((tensor*) t->parent_l.value.t, topo, seen, n, ns);
	if (t->parent_r.type == TENSOR) topo_dfs((tensor*) t->parent_r.value.t, topo, seen, n, ns);
	topo[(*n)++]=t;
}

void try_init_parent_grad(tensor_parent *parent) {
	if (parent->type != TENSOR) return;
	tensor *t = (tensor*)parent->value.t;
	if (t->grad != NULL) return;
	t->grad = alloc_tensor(t->shape, t->ndims, 0, NEW, t->is_param);
}

// Broadcasts reduced tensor to match parent shape for backprop
tensor *_unreduce_tensor(tensor *from, tensor *node, tensor *parent) {
	tensor *broadcast_g = alloc_tensor(node->shape, node->ndims, 0, NEW, false);
	copy_data(broadcast_g, from);

	tensor *t = alloc_tensor(parent->shape, parent->ndims, 0, NEW, false);
	i32 axis = node->grad_arg;

	if (! node->grad_keptdims) {
		i32 *nshape = (i32*)arena_alloc(parent->ndims * sizeof(i32));
		nshape[axis]=1;
		for (int i=0; i<axis; ++i) nshape[i]=parent->shape[i];
		for (int i=axis+1; i<parent->ndims; ++i) nshape[i]=parent->shape[i];
		broadcast_g = tensor_reshape(broadcast_g, nshape, parent->ndims);
	}

	i32 *bstrides = broadcast_strides(broadcast_g);
	i32 *indices = (i32*)arena_alloc(parent->ndims * sizeof(i32));
	memset(indices, 0, parent->ndims * sizeof(i32));
	do {
		f32 *pv = tensor_getitem(t, parent->strides, indices);
		*pv = *tensor_getitem(broadcast_g, bstrides, indices);
	} while (inc_shapeindex(indices, parent->shape, parent->ndims) != -1);
	return t;
}

tensor *_unbroadcast_grad(tensor *g, tensor *parent) {
	for (int i=0; i<parent->ndims; i++)
		if (parent->shape[i] == 1 && g->shape[i] > 1)
			g = tensor_sum(g, i, true);
	return g;
}

tensor *tensor_backward(tensor *t) {
	tensor *topo[MAX_TOPO_NODES], *seen[MAX_TOPO_NODES];
	i32 n =0, ns = 0;
	topo_dfs(t, topo, seen, &n, &ns);

	t->grad = alloc_tensor(t->shape, t->ndims, 1.0f, NEW, t->is_param);
	for (i32 i=n-1; i>0; i--) {
		tensor *node = topo[i];
		tensor *g = (tensor*)node->grad;

		try_init_parent_grad(&node->parent_r);
		try_init_parent_grad(&node->parent_l);

		tensor *lp, *rp, *lg, *rg, *dl, *dr;

		if (node->parent_l.type == TENSOR) {
			lp = (tensor*)node->parent_l.value.t;
			lg = (tensor*)lp->grad;
		}
		if (node->parent_r.type == TENSOR) {
			rp = (tensor*)node->parent_r.value.t;
			rg = (tensor*)rp->grad;
		}

		switch (node->parent_op) {
			case NEW: continue; break;
			case RESHAPE: dl = tensor_reshape(g, lp->shape, lp->ndims); break;
			case POW: dl = tensor_mul(g, tensor_mul_scalar(tensor_pow(lp, node->grad_arg-1), node->grad_arg)); break;
			case EXP: dl = tensor_mul(g, node); break;
			case LOG: dl = tensor_mul(g, tensor_pow(lp, -1)); break;
			case ADD:
				dl = _unbroadcast_grad(g, lp);
				if (node->parent_r.type == TENSOR) dr = _unbroadcast_grad(g, rp);
				break;	
			case MUL:
				if (node->parent_r.type == TENSOR) {
					dl = _unbroadcast_grad(tensor_mul(g, rp), lp);
					dr = _unbroadcast_grad(tensor_mul(g, lp), rp);
				} else {
					dl = _unbroadcast_grad(tensor_mul_scalar(g, node->parent_r.value.s), lp);
				}
				break;
			case SUM: dl = _unreduce_tensor(g, node, lp); break;
			case MAX: dl = tensor_mul(_unreduce_tensor(g, node, lp), tensor_eq(_unreduce_tensor(node, node, lp), lp)); break;
			case RELU: dl = tensor_mul(g, _tensor_reluback(lp)); break;
			default: continue; break;
		}
		if (node->parent_l.type == TENSOR) copy_data(lg, tensor_add(lg, dl));
		if (node->parent_r.type == TENSOR) copy_data(rg, tensor_add(rg, dr));
	}
	arena_clear();
	return topo[n];
}

void init_tensor(tensor *t, i32 *shape, i32 ndims, f32 init, tensor_op op, bool is_param) {
	i32 *nshape, *strides; f32 *data;
	i32 size = get_size(shape, ndims);
	void *(*alloc)(size_t) = is_param ? &malloc : &arena_alloc;

	nshape = (i32*) alloc(ndims * sizeof(i32));
	strides = (i32*) alloc(ndims * sizeof(i32));
	data = (f32*) alloc(size * sizeof(f32));

	calculate_strides(shape, ndims, &strides);
	for (int i=0; i<size; ++i) data[i]=init;
	memcpy(nshape, shape, ndims * sizeof(i32));

	t->parent_op = op;
	t->parent_l.type = NONE; t->parent_r.type = NONE;
	t->ndims = ndims; t->is_param = is_param;
	t->data = data; t->grad = NULL;
	t->shape = nshape; t->strides = strides;
}

tensor *alloc_tensor(i32 *shape, i32 ndims, f32 init, tensor_op op, bool is_param) {
	tensor *t = is_param ? (tensor*)malloc(sizeof(tensor)) : (tensor*)arena_alloc(sizeof(tensor));
	init_tensor(t, shape, ndims, init, op, is_param);
	return t;
}

typedef struct {
	tensor *weights, *bias;
	i32 inputs, outputs;
} layer_linear;

layer_linear *linear_init(i32 inputs, i32 outputs) {
	layer_linear *layer = (layer_linear*)malloc(sizeof(layer_linear));
	layer->inputs = inputs; layer->outputs = outputs;
	// TODO: heuristic weight init values
	i32 wshape[2] = { outputs, inputs }; layer->weights = alloc_tensor(wshape, 2, 0, NEW, true);
	i32 bshape[2] = { outputs, 1 }; layer->bias = alloc_tensor(bshape, 2, 0, NEW, true);
	return layer;
}
tensor *linear_forward(layer_linear *layer, tensor *x) { return tensor_add(tensor_gemm(x, layer->weights), layer->bias); }

tensor *tensor_pad(tensor *t, i32 ph, i32 pw) {
	i32 *nshape = (i32*)arena_alloc(t->ndims * sizeof(i32));
	nshape[0]=t->shape[0]; nshape[3]=t->shape[3];
	nshape[1]=t->shape[1] + ph * 2; nshape[2]=t->shape[2] + pw * 2;
	// TODO: Copy autograd info??
	tensor *pt = alloc_tensor(nshape, t->ndims, 0, NEW, false);
	int indices[4] = { 0, 0, 0, 0 }; int pindices[4];
	do {
		memcpy(pindices, indices, 4 * sizeof(i32));
		pindices[1]+=ph; pindices[2]+=pw;
		f32 *c = tensor_getitem(pt, pt->strides, pindices);
		*c = *tensor_getitem(t, t->strides, indices);
	} while(inc_shapeindex(indices, t->shape, t->ndims) != -1);
	return pt;
}

tensor *tensor_im2col(tensor *im, i32 kh, i32 kw, i32 sh, i32 sw, i32 ph, i32 pw) {
	i32 N = im->shape[0], h = im->shape[1], w=im->shape[2], c = im->shape[3];
	i32 oh = floor((h - kh + ph*2) / sh) + 1, ow = floor((w - kw + pw*2) / sw) + 1;
	im = tensor_pad(im, ph, pw);
	i32 cshape[2] = { N * oh * ow, c * kh * kw};
	tensor *cols = alloc_tensor(cshape, 2,0, NEW, false);

	// TODO: parallelize?
	i32 row=0;
	for (int n=0; n<N; ++n) {// Iterate over # of distinct kernel windows
		for (int j=0; j<oh; ++j) {
			for (int k=0; k<ow; ++k) {// Copy current kernel window into col
				i32 col=0;
				for (int kj=0; kj<kh; ++kj) {
					for (int kk=0; kk<kw; ++kk) {
						for (int cc=0; cc<c; ++cc) {
							i32 iter[4] = { n, kj+j*sh, kk+k*sw, cc };
							cols->data[row*cshape[1] + col]=*tensor_getitem(im, im->strides, iter);
							col++;
						}
					}
				}
				row++;
			}
		}
	}
	return cols;
}

typedef struct {
	tensor *weights, *bias;
	i32 kstrides, ksize, padding, channels_in, channels_out;
} layer_conv2d;

layer_conv2d *conv2d_init(i32 channels_in, i32 channels_out, i32 kstrides, i32 ksize, i32 padding) {
	layer_conv2d *layer = (layer_conv2d*)malloc(sizeof(layer_conv2d));
	layer->kstrides=kstrides; layer->ksize=ksize; layer->padding=padding;
	layer->channels_in=channels_in; layer->channels_out=channels_out;
	i32 wshape[4] = { ksize, ksize, channels_in, channels_out }; layer->weights = alloc_tensor(wshape, 4, 0, NEW, true);
	i32 bshape[1] = { 1 }; layer->bias = alloc_tensor(bshape, 1, 0, NEW, true);
	return layer;
}
tensor *conv2d_forward(layer_conv2d *layer, tensor *x) {
	tensor *cols = tensor_im2col(x, layer->ksize, layer->ksize, layer->kstrides, layer->kstrides, layer->padding, layer->padding);
	i32 kshape[2] = { layer->ksize * layer->ksize * x->shape[3], layer->channels_out };
	tensor *kernel = tensor_reshape(layer->weights, kshape, 2); 
	return tensor_add(tensor_gemm(cols, kernel), layer->bias);
}

typedef struct {
	tensor *weights, *bias;
	i32 channels;
} layer_batchnorm;

layer_batchnorm *batchnorm_init(i32 channels) {
	layer_batchnorm *layer = (layer_batchnorm*)malloc(sizeof(layer_batchnorm));
	i32 pshape[2] = { channels, 1 };
	layer->channels = channels;
	layer->weights = alloc_tensor(pshape, 2, 1, NEW, true);
	layer->bias = alloc_tensor(pshape, 2, 1, NEW, true);
	return layer;
}

tensor *batchnorm_forward(layer_batchnorm *layer, tensor *x) {
	tensor *mean = x; tensor *var;
	for (i32 i=0; i<x->ndims-1; ++i) mean = tensor_mean(mean, i, true);
	var = tensor_pow(tensor_sub(x, mean), 2.0);
	for (i32 i=0; i<x->ndims-1; ++i) var = tensor_mean(var, i, true);
	x = tensor_div(tensor_sub(x, mean), tensor_sqrt(tensor_add_scalar(var, 1e-6)));
	return tensor_add(tensor_mul(x, layer->weights), layer->bias);
}

typedef struct {
	tensor **m, **v, **params;
	i32 nparams, step;
	f32 step_size, b1, b2;
} optim_ADAM;

optim_ADAM *ADAM_init(tensor **params, i32 nparams, f32 step_size, f32 b1, f32 b2) {
	optim_ADAM *adam = (optim_ADAM*)malloc(sizeof(optim_ADAM));
	tensor **m = (tensor**)calloc(nparams, sizeof(tensor)), **v = (tensor**)calloc(nparams, sizeof(tensor));
	for (i32 i=0; i<nparams; ++i) {
		m[i] = alloc_tensor(params[i]->shape, params[i]->ndims, 0, NEW, true);
		v[i] = alloc_tensor(params[i]->shape, params[i]->ndims, 0, NEW, true);
	}
	adam->params = params; adam->nparams = nparams;
	adam->step_size = step_size; adam->b1 = b1; adam->b2 = b2;
	adam->step = 0;
	return adam;
}

void ADAM_step(optim_ADAM *adam) {
	adam->step++;
	for (int i=0; i<adam->nparams; ++i) {
		tensor *g = (tensor*)adam->params[i]->grad;
		tensor *m =tensor_add(tensor_mul_scalar(adam->m[i], adam->b1), tensor_mul_scalar(g, 1 - adam->b1));
		tensor *v =tensor_add(tensor_mul_scalar(adam->v[i], adam->b2), tensor_mul_scalar(g, 1 - adam->b2));
		tensor *m_hat = tensor_mul_scalar(m, 1.0 / (1.0 - powf(adam->b1, adam->step)));
		tensor *v_hat = tensor_mul_scalar(v, 1.0 / (1.0 - powf(adam->b2, adam->step)));
		tensor *out = tensor_sub(adam->params[i], tensor_mul_scalar(tensor_div(m_hat, tensor_add_scalar(tensor_sqrt(v_hat), 1e-6)), adam->step_size));
		copy_data(adam->m[i], m);
		copy_data(adam->v[i], v);
		copy_data(adam->params[i], out);
	}
}
