#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <omp.h>

#define MAX_TOPO_NODES 1024

// Profiling
static double prof_gemm=0, prof_im2col=0, prof_col2im=0, prof_reduce=0, prof_unreduce=0, prof_biop=0, prof_unop=0;
void prof_print() {
	printf("PROFILE: gemm=%.3f im2col=%.3f col2im=%.3f reduce=%.3f unreduce=%.3f biop=%.3f unop=%.3f\n",
		prof_gemm, prof_im2col, prof_col2im, prof_reduce, prof_unreduce, prof_biop, prof_unop);
}
void prof_reset() { prof_gemm=prof_im2col=prof_col2im=prof_reduce=prof_unreduce=prof_biop=prof_unop=0; }
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
	NEW, RESHAPE,
	POW, EXP, LOG,
	MUL, ADD,
	SUM, MAX,
	GEMM, RELU,
	IM2COL
} tensor_op;

const char* const op_names[] = {
    "NEW", "RESHAPE", "POW", "EXP", "LOG", "MUL",
		"ADD", "SUM", "MAX", "GEMM", "RELU", "IM2COL"
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
	f32 grad_arg; i32 *col2im_args;
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
tensor *tensor_im2col(tensor *im, i32 ksize, i32 strides);
tensor *tensor_col2im(tensor *cols, tensor *meta);

// Helper funcs
i32 get_size(i32 *shape, i32 ndims) {
	i32 size = 1;
	for (i32 i=0; i<ndims; ++i) size *= shape[i];
	return size;
}

void copy_data(tensor *dst, tensor *src) { memcpy(dst->data, src->data, get_size(dst->shape, dst->ndims) * sizeof(f32)); }

void tensor_accumulate(tensor *dst, tensor *src) {
	#pragma omp parallel for
	for (i32 i=0; i<get_size(dst->shape, dst->ndims); ++i)
		dst->data[i]+=src->data[i];
}

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

tensor *tensor_transpose2d(tensor *t) {
	i32 nshape[2] = { t->shape[1], t->shape[0] };
	return tensor_reshape(t, nshape, 2);
}

tensor *tensor_reshape(tensor *t, i32 *newshape, i32 ndims) {
	if (get_size(newshape, ndims) != get_size(t->shape, t->ndims)) return NULL;

	tensor *nt = alloc_tensor(newshape, ndims, 0, RESHAPE, false);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	memcpy(nt->data, t->data, get_size(t->shape, t->ndims) * sizeof(f32));
	return nt;
}

tensor *tensor_apply_unop(tensor *t, f32 (*func)(f32, f32), f32 arg, tensor_op op) {
	double _t0 = omp_get_wtime();
	tensor *nt = alloc_tensor(t->shape, t->ndims, 0, op, false);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	nt->parent_r.type = NONE; nt->grad_arg = arg;

	#pragma omp parallel for
	for (i32 i=0; i<get_size(t->shape, t->ndims); ++i)
		nt->data[i]=func(t->data[i], arg);
	prof_unop += omp_get_wtime() - _t0;
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
	double _t0 = omp_get_wtime();
	if (a->ndims != b->ndims) return NULL;
	i32 *cshape = broadcast_shape(a->shape, b->shape, a->ndims);
	tensor *c = alloc_tensor(cshape, a->ndims, 0, op, false);

	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = TENSOR; c->parent_r.value.t = b;

	bool same_shape = true;
	for (i32 i=0; i<a->ndims; ++i) if (a->shape[i] != b->shape[i]) same_shape=false;

	i32 M=c->shape[0], N=c->shape[1];
	if (same_shape) {
		#pragma omp parallel for
		for (i32 i=0; i<get_size(a->shape, a->ndims); ++i)
			c->data[i]=func(a->data[i], b->data[i]);
	} else if (a->shape[0] == 1 || b->shape[0] == 1) {
		f32 *row = a->shape[0] == 1 ? a->data : b->data;
		f32 *big = a->shape[0] == 1 ? b->data : a->data;
		#pragma omp parallel for
		for (i32 i=0; i<M; ++i)
			for (i32 j=0; j<N; ++j)
				c->data[i*N + j]=func(row[j], big[i*N + j]);
	} else if (a->shape[1] == 1 || b->shape[1] == 1) {
		f32 *col = a->shape[1] == 1 ? a->data : b->data;
		f32 *big = a->shape[1] == 1 ? b->data : a->data;
		#pragma omp parallel for
		for (i32 i=0; i<M; ++i)
			for (i32 j=0; j<N; ++j)
				c->data[i*N + j]=func(col[i], big[i*N + j]);
	} else {
		printf("Invalid elemwise op\n");
		exit(0);
	}

	prof_biop += omp_get_wtime() - _t0;
	return c;
}

tensor *tensor_apply_biop_scalar(tensor *a, f32 b, f32 (*func)(f32, f32), tensor_op op) {
	tensor *c = alloc_tensor(a->shape, a->ndims, 0, op, false);
	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = SCALAR; c->parent_r.value.s = b;

	#pragma omp parallel for
	for (i32 i=0; i<get_size(a->shape, a->ndims); ++i) 
		c->data[i]=func(a->data[i], b);

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
	double _t0 = omp_get_wtime();
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


	if (t->ndims == 2) {
		if (axis == 1) {
			#pragma omp parallel for
			for (i32 i=0; i<t->shape[0]; ++i)
				for (i32 j=0; j<t->shape[1]; ++j)
					r->data[i]+=t->data[i*t->shape[1] + j];
		} else {
			// Prevent race conditions
			#pragma omp parallel
			{
				f32 *local = (f32*)calloc(t->shape[1], sizeof(f32));
				#pragma omp for
				for (i32 i=0; i<t->shape[0]; ++i)
					for (i32 j=0; j<t->shape[1]; ++j)
						local[j]+=t->data[i*t->shape[1] + j];
				#pragma omp critical
				for (i32 i=0; i<t->shape[1]; ++i) r->data[i] += local[i];
				free(local);
			}
		}
	} else {
		i32 *riter = (i32*)arena_alloc(r->ndims * sizeof(i32)), *iter = (i32*)arena_alloc(t->ndims * sizeof(i32));
		memset(riter, 0, r->ndims * sizeof(i32));
		// TODO: parallelize?
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
	}
	prof_reduce += omp_get_wtime() - _t0;
	return r;
}

void __sum(f32 *a, f32 b) { *a += b; }
void __max(f32 *a, f32 b) { *a = *a > b ? *a : b; }
tensor *tensor_sum(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__sum, 0, SUM); }
tensor *tensor_max(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__max, -INFINITY, MAX); }
tensor *tensor_mean(tensor *t, i32 axis, bool keepdims) { return tensor_mul_scalar(tensor_sum(t, axis, keepdims), 1.0 / (f32)t->shape[axis]); }

tensor *tensor_gemm(tensor *a, tensor *b) {
	double _t0 = omp_get_wtime();
	if (a->shape[1] != b->shape[0]) return NULL;
	i32 M=a->shape[0], K=a->shape[1], N=b->shape[1];
	i32 cshape[2] = { M, N };
	tensor *c = alloc_tensor(cshape, 2, 0, GEMM, false);
	c->parent_l.type = TENSOR; c->parent_l.value.t=a;
	c->parent_r.type = TENSOR; c->parent_r.value.t=b;

	#pragma omp parallel for
	for (int i=0; i<M; ++i) {
		for (int k=0; k<K; ++k) {
			f32 a_val = a->data[i*K + k];
			for (int j=0; j<N; ++j)
				c->data[i*N + j]+=a_val*b->data[k*N + j];
		}
	}
	prof_gemm += omp_get_wtime() - _t0;
	return c;
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
	double _t0 = omp_get_wtime();
	tensor *broadcast_g = alloc_tensor(node->shape, node->ndims, 0, NEW, false);
	copy_data(broadcast_g, from);

	tensor *t = alloc_tensor(parent->shape, parent->ndims, 0, NEW, false);
	i32 axis = (i32)node->grad_arg;

	if (! node->grad_keptdims) {
		i32 *nshape = (i32*)arena_alloc(parent->ndims * sizeof(i32));
		nshape[axis]=1;
		for (int i=0; i<axis; ++i) nshape[i]=parent->shape[i];
		for (int i=axis+1; i<parent->ndims; ++i) nshape[i]=parent->shape[i];
		broadcast_g = tensor_reshape(broadcast_g, nshape, parent->ndims);
	}

	if (parent->ndims == 2) {
		i32 M = parent->shape[0], N = parent->shape[1];
		if (axis == 1) {
			#pragma omp parallel for
			for (i32 i=0; i<M; ++i)
				for (i32 j=0; j<N; ++j)
					t->data[i*N + j] = broadcast_g->data[i];
		} else {
			#pragma omp parallel for
			for (i32 i=0; i<M; ++i)
				for (i32 j=0; j<N; ++j)
					t->data[i*N + j] = broadcast_g->data[j];
		}
	} else {
		i32 *bstrides = broadcast_strides(broadcast_g);
		i32 *indices = (i32*)arena_alloc(parent->ndims * sizeof(i32));
		memset(indices, 0, parent->ndims * sizeof(i32));
		do {
			f32 *pv = tensor_getitem(t, parent->strides, indices);
			*pv = *tensor_getitem(broadcast_g, bstrides, indices);
		} while (inc_shapeindex(indices, parent->shape, parent->ndims) != -1);
	}
	prof_unreduce += omp_get_wtime() - _t0;
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

	if (t->grad == NULL) {
		t->grad = alloc_tensor(t->shape, t->ndims, 1.0f, NEW, t->is_param);
	} else {
		i32 size = get_size(t->shape, t->ndims);
		f32 *d = ((tensor*)t->grad)->data;
		for (i32 i = 0; i < size; ++i) d[i] = 1.0f;
	}
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
			case NEW: break;
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
			case IM2COL: dl = tensor_col2im(g, node); break;
			case GEMM:
				dl = tensor_gemm(g, tensor_transpose2d(rp));
				dr = tensor_gemm(tensor_transpose2d(lp), g);
				break;
			default: continue; break;
		}
		if (node->parent_l.type == TENSOR) tensor_accumulate(lg, dl);
		if (node->parent_r.type == TENSOR) tensor_accumulate(rg, dr);
	}
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
	if (init == 0.0f) memset(data, 0.0f, size * sizeof(f32));
	else for (int i=0; i<size; ++i) data[i]=init;
	memcpy(nshape, shape, ndims * sizeof(i32));

	t->parent_op = op;
	t->parent_l.type = NONE; t->parent_r.type = NONE;
	t->ndims = ndims; t->is_param = is_param;
	t->data = data; t->grad = NULL;
	t->shape = nshape; t->strides = strides;
	t->col2im_args = (i32*)alloc(8 * sizeof(i32));
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

f32 box_mueller_rand() {
	float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
	float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

void init_he(tensor *weights, i32 n) {
	f32 std = sqrt(2.0 / n);
	for (int i=0; i<n; ++i) weights->data[i]=box_mueller_rand() * std;
}


layer_linear *linear_init(i32 inputs, i32 outputs) {
	layer_linear *layer = (layer_linear*)malloc(sizeof(layer_linear));
	layer->inputs = inputs; layer->outputs = outputs;
	i32 wshape[2] = { inputs, outputs }; layer->weights = alloc_tensor(wshape, 2, 0, NEW, true);
	init_he(layer->weights, outputs * inputs);
	i32 bshape[2] = { 1, outputs }; layer->bias = alloc_tensor(bshape, 2, 0, NEW, true);
	return layer;
}
tensor *linear_forward(layer_linear *layer, tensor *x) {
	return tensor_add(tensor_gemm(x, layer->weights), layer->bias);
}

tensor *tensor_im2col(tensor *im, i32 ksize, i32 strides) {
	double _t0 = omp_get_wtime();
	i32 N = im->shape[0], h = im->shape[1], w=im->shape[2], c = im->shape[3];
	i32 oh = floor((h - ksize) / strides) + 1, ow = floor((w - ksize) / strides) + 1;
	i32 cshape[2] = { N * oh * ow, c * ksize*ksize };
	tensor *cols = alloc_tensor(cshape, 2,0, IM2COL, false);
	cols->parent_l.type = TENSOR; cols->parent_l.value.t=im;
	cols->col2im_args[0] = ksize; cols->col2im_args[1] = strides;
	cols->col2im_args[2] = oh; cols->col2im_args[3] = ow;
	cols->col2im_args[4] = N; cols->col2im_args[5]=c;
	cols->col2im_args[6] = h; cols->col2im_args[7]=w;

	#pragma omp parallel for collapse(3)
	for (int n=0; n<N; ++n) {// Iterate over # of distinct kernel windows
		for (int j=0; j<oh; ++j) {
			for (int k=0; k<ow; ++k) {// Copy current kernel window into col
				i32 row=n*oh*ow +j*ow + k;
				i32 col=0;
				i32 im_base = n*h*w*c;
				for (int kj=0; kj<ksize; ++kj) {
					i32 im_row = (kj + j*strides) * w * c;
					for (int kk=0; kk<ksize; ++kk) {
						i32 im_off = im_base + im_row + (kk + k*strides) * c;
						for (int cc=0; cc<c; ++cc)
							cols->data[row*cshape[1] + col++]=im->data[im_off + cc];
					}
				}
			}
		}
	}
	prof_im2col += omp_get_wtime() - _t0;
	return cols;
}

// Transform (N*oh*ow, kh*kw*c) -> (N, h, w, c)
tensor *tensor_col2im(tensor *cols, tensor *meta) {
	double _t0 = omp_get_wtime();
	i32 N=meta->col2im_args[4], oh=meta->col2im_args[2], ow=meta->col2im_args[3];
	i32 ksize=meta->col2im_args[0], strides=meta->col2im_args[1], c=meta->col2im_args[5];
	i32 h=meta->col2im_args[6], w=meta->col2im_args[7];

	i32 imshape[4] = { N, h, w, c };
	tensor *im = alloc_tensor(imshape, 4, 0, NEW, false);
	i32 col_width = ksize*ksize*c;

	#pragma omp parallel for
	for (int n=0; n<N; ++n) {
		i32 off_base = n*h*w*c;
		for (int j=0; j<oh; ++j) {
			for (int k=0; k<ow; ++k) {
				i32 row =n*ow*ow + j*ow + k;
				for (int kj=0; kj<ksize; ++kj) {
					i32 off_row = (kj + j*strides) * w * c;
					for (int kk=0; kk<ksize; ++kk) {
						i32 off = off_base + off_row + (kk * k*strides) * c;
						for (int cc=0; cc<c; ++cc)
							im->data[off+cc] += cols->data[row*col_width + (kj*ksize + kk)*c + cc];
					}
				}
			}
		}
	}
	prof_col2im += omp_get_wtime() - _t0;
	return im;
}

tensor *tensor_maxpool2d(tensor *t, i32 psize, i32 strides) {
	tensor *cols = tensor_im2col(t, psize, strides);
	i32 N=t->shape[0], c=t->shape[3];
	i32 oh=floor((t->shape[1]-psize) / strides)+1, ow=floor((t->shape[2]-psize) / strides)+1;
	i32 sep[3] = { N*oh*ow, psize*psize, c };
	i32 oshape[4] = { N, oh, ow, c };
	cols = tensor_reshape(cols, sep, 3);
	cols = tensor_max(cols, 1, true);
	return tensor_reshape(cols, oshape, 4);
}

typedef struct {
	tensor *weights, *bias;
	i32 kstrides, ksize, channels_in, channels_out;
} layer_conv2d;

layer_conv2d *conv2d_init(i32 channels_in, i32 channels_out, i32 kstrides, i32 ksize) {
	layer_conv2d *layer = (layer_conv2d*)malloc(sizeof(layer_conv2d));
	layer->kstrides=kstrides; layer->ksize=ksize;
	layer->channels_in=channels_in; layer->channels_out=channels_out;
	i32 wshape[4] = { ksize, ksize, channels_in, channels_out }; layer->weights = alloc_tensor(wshape, 4, 0, NEW, true);
	i32 bshape[2] = { 1, channels_out }; layer->bias = alloc_tensor(bshape, 2, 0, NEW, true);
	return layer;
}
tensor *conv2d_forward(layer_conv2d *layer, tensor *x) {
	tensor *cols = tensor_im2col(x, layer->ksize, layer->kstrides);
	i32 kshape[2] = { layer->ksize * layer->ksize * x->shape[3], layer->channels_out };
	i32 oh=floor((x->shape[1]-layer->ksize) / layer->kstrides)+1, ow=floor((x->shape[2]-layer->ksize) / layer->kstrides)+1;
	i32 oshape[4]={x->shape[0], oh, ow, layer->channels_out };
	tensor *kernel = tensor_reshape(layer->weights, kshape, 2); 
	tensor *out = tensor_add(tensor_gemm(cols, kernel), layer->bias);
	return tensor_reshape(out, oshape, 4);
}

typedef struct {
	tensor *weights, *bias;
	i32 channels;
} layer_batchnorm;

layer_batchnorm *batchnorm_init(i32 channels) {
	layer_batchnorm *layer = (layer_batchnorm*)malloc(sizeof(layer_batchnorm));
	i32 pshape[2] = { 1, channels };
	layer->channels = channels;
	layer->weights = alloc_tensor(pshape, 2, 1, NEW, true);
	layer->bias = alloc_tensor(pshape, 2, 1, NEW, true);
	return layer;
}

tensor *batchnorm_forward(layer_batchnorm *layer, tensor *x) {
	i32 *orig_shape = x->shape; i32 orig_ndims = x->ndims;
	i32 flat[2] = { get_size(x->shape, x->ndims-1), x->shape[3] };
	x = tensor_reshape(x, flat, 2);

	tensor *mean = tensor_mean(x, 0, true);
	tensor *var = tensor_mean(tensor_pow(tensor_sub(x, mean), 2.0), 0, true);
	x = tensor_div(tensor_sub(x, mean), tensor_sqrt(tensor_add_scalar(var, 1e-6)));
	x = tensor_add(tensor_mul(x, layer->weights), layer->bias);

	return tensor_reshape(x, orig_shape, orig_ndims);
}

void zero_grads(tensor **params, i32 nparams) {
	for (i32 i=0; i<nparams; ++i)
		if (params[i]->grad != NULL)
			memset(((tensor*)params[i]->grad)->data, 0.0f, get_size(params[i]->shape, params[i]->ndims) * sizeof(f32));
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
		params[i]->grad = alloc_tensor(params[i]->shape, params[i]->ndims, 0, NEW, true);
		m[i] = alloc_tensor(params[i]->shape, params[i]->ndims, 0, NEW, true);
		v[i] = alloc_tensor(params[i]->shape, params[i]->ndims, 0, NEW, true);
	}
	adam->m=m; adam->v=v;
	adam->params = (tensor**)malloc(nparams * sizeof(tensor*));
	memcpy(adam->params, params, nparams * sizeof(tensor*));
	adam->nparams = nparams;
	adam->step_size = step_size; adam->b1 = b1; adam->b2 = b2;
	adam->step = 0;
	return adam;
}

void ADAM_step(optim_ADAM *adam) {
	adam->step++;
	f32 bc1 = 1.0f - powf(adam->b1, adam->step);
	f32 bc2 = 1.0f - powf(adam->b2, adam->step);
	for (int i=0; i<adam->nparams; ++i) {
		if (adam->params[i]== NULL || !adam->params[i]->grad) {
			printf("ADAM found dirty grad.\n");
			continue;
		}
		tensor *p=adam->params[i], *g = (tensor*)p->grad;
		i32 size = get_size(p->shape, p->ndims);
		#pragma omp parallel for
		for (i32 j=0; j<size; ++j) {
			adam->m[i]->data[j] = adam->b1 * adam->m[i]->data[j] + (1-adam->b1) * g->data[j];
			adam->v[i]->data[j] = adam->b2 * adam->v[i]->data[j] + (1-adam->b2) * g->data[j] * g->data[j];
			f32 mh = adam->m[i]->data[j] / bc1;
			f32 vh = adam->v[i]->data[j] / bc2;
			p->data[j] -= adam->step_size * mh / (sqrtf(vh) + 1e-6f);
		}
	}
}
