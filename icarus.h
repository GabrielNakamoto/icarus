#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOPO_NODES 1024

typedef float f32;
typedef int i32;

typedef enum {
	NEW,
	RESHAPE,
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
	bool grad_keptdims;
	void *grad; // tensor
	bool free_after_grad;

	// Metadata
	i32 ndims;
	i32 *shape, *strides;

	f32 *data;
} tensor;


tensor *alloc_tensor(i32 *shape, i32 ndims, f32 init, tensor_op op);
void free_tensor(tensor *t);
tensor *duplicate_tensor(tensor *t);
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
tensor *tensor_div(tensor *a, tensor *b);
tensor *tensor_add(tensor *a, tensor *b);
tensor *tensor_sub(tensor *a, tensor *b);

tensor *tensor_mul_scalar(tensor *a, f32 b);
tensor *tensor_div_scalar(tensor *a, f32 b);
tensor *tensor_add_scalar(tensor *a, f32 b);
tensor *tensor_sub_scalar(tensor *a, f32 b);

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

void print_shape(tensor *t) {
	printf("(");
	for (int i=0; i<t->ndims; ++i) {
		printf("%d", t->shape[i]);
		if (i<t->ndims-1) printf(",");
	}
	printf(")\n");
}

void print_2d_tensor(tensor *t) {
	printf("Op->%s\n", op_names[t->parent_op]);
	printf("Left Parent->%s\n", parent_type_names[t->parent_l.type]);
	printf("Right Parent->%s\n", parent_type_names[t->parent_r.type]);
	for (int i=0; i<t->shape[0]; ++i) {
		for (int j=0; j<t->shape[1]; ++j) printf("%.4f ", t->data[j + i*t->shape[1]]);
		printf("\n");
	}
	printf("\n");
}


// Helper funcs
static i32 get_size(i32 *shape, i32 ndims) {
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
	i32 *bstrides = (i32*) malloc(t->ndims * sizeof(i32));
	for (i32 i=0; i<t->ndims; ++i) bstrides[i] = t->shape[i] == 1 ? 0 : t->strides[i];
	return bstrides;
}

i32 *broadcast_shape(i32 *a_sh, i32 *b_sh, i32 ndims) {
	i32 *bshape = (i32*) malloc(ndims * sizeof(i32));
	for (i32 i=0; i<ndims; ++i) bshape[i] = a_sh[i] == 1 ? b_sh[i] : a_sh[i];
	return bshape;
}

i32 tensor_getidx(i32 *strides, i32 ndims, i32 *indices) {
	i32 idx = 0;
	for (i32 i=0; i<ndims; ++i) idx += strides[i] * indices[i];
	return idx;
}

f32 *tensor_getitem(tensor *t, i32 *strides, i32 *indices) {
	return t->data + tensor_getidx(strides, t->ndims, indices);
}

i32 inc_shapeindex(i32 *indices, i32 *shape, i32 ndims) {
	indices[ndims-1]++;
	for (i32 i=ndims-1; i>=0; i--) {
		if (indices[i] == shape[i]) {
			indices[i]=0;
			if (i == 0) return -1;
			indices[i-1]++;
		} else {
			break;
		}
	}
	return 1;
}

tensor *tensor_reshape(tensor *t, i32 *newshape, i32 ndims) {
	if (get_size(newshape, ndims) != get_size(t->shape, t->ndims)) return NULL;

	tensor *nt = alloc_tensor(newshape, ndims, 0, RESHAPE);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	memcpy(nt->data, t->data, get_size(t->shape, t->ndims) * sizeof(f32));
	return nt;
}

tensor *tensor_apply_unop(tensor *t, f32 (*func)(f32, f32), f32 arg, tensor_op op) {
	tensor *nt = alloc_tensor(t->shape, t->ndims, 0, op);
	nt->parent_l.type = TENSOR; nt->parent_l.value.t = t;
	nt->parent_r.type = NONE; nt->grad_arg = arg;

	i32 *iter = (i32*) malloc(t->ndims * sizeof(i32));
	memset(iter, 0, t->ndims * sizeof(i32));
	do {
		f32 *b = tensor_getitem(nt, nt->strides, iter);
		*b = func(*tensor_getitem(t, t->strides, iter), arg);
	} while(inc_shapeindex(iter, t->shape, t->ndims) != -1);
	free(iter);
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
	tensor *c = alloc_tensor(cshape, a->ndims, 0, op);
	free(cshape);

	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = TENSOR; c->parent_r.value.t = b;

	i32 *a_bstrides = broadcast_strides(a); i32 *b_bstrides = broadcast_strides(b);
	i32 *iter = (i32*) malloc(a->ndims * sizeof(i32));
	memset(iter, 0, a->ndims * sizeof(i32));
	do {
		f32 *cv = tensor_getitem(c, c->strides, iter);
		*cv = func(*tensor_getitem(a, a_bstrides, iter), *tensor_getitem(b, b_bstrides, iter));
	} while (inc_shapeindex(iter, c->shape, c->ndims) != -1);
	free(iter);

	return c;
}

tensor *tensor_apply_biop_scalar(tensor *a, f32 b, f32 (*func)(f32, f32), tensor_op op) {
	tensor *c = alloc_tensor(a->shape, a->ndims, 0, op);

	i32 *a_bstrides = broadcast_strides(a);
	i32 *iter = (i32*) malloc(a->ndims * sizeof(i32));
	memset(iter, 0, a->ndims * sizeof(i32));

	c->parent_l.type = TENSOR; c->parent_l.value.t = a;
	c->parent_r.type = SCALAR; c->parent_r.value.s = b;

	do {
		f32 *cv = tensor_getitem(c, c->strides, iter);
		*cv = func(*tensor_getitem(a, a_bstrides, iter), b);
	} while (inc_shapeindex(iter, c->shape, c->ndims) != -1);
	free(iter);

	return c;
}

f32 __mul(f32 a, f32 b) { return a * b; }
f32 __add(f32 a, f32 b) { return a + b; }
f32 __eq(f32 a, f32 b) { return a == b ? 1 : 0; }
tensor *tensor_mul(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__mul, MUL); }
tensor *tensor_div(tensor *a, tensor *b) {
	tensor *denom = tensor_pow(b, -1); denom->free_after_grad = true;
	return tensor_mul(a, denom);
}
tensor *tensor_add(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__add, ADD); }
tensor *tensor_sub(tensor *a, tensor *b) {
	tensor *neg = tensor_mul_scalar(b, -1); neg->free_after_grad = true;
	return tensor_add(a, neg);
}
tensor *tensor_eq(tensor *a, tensor *b) { return tensor_apply_biop(a, b, &__eq, NEW); }
tensor *tensor_mul_scalar(tensor *a, f32 b) { return tensor_apply_biop_scalar(a, b, &__mul, MUL); }
tensor *tensor_div_scalar(tensor *a, f32 b) { return tensor_mul_scalar(a, 1.0 / b); }
tensor *tensor_add_scalar(tensor *a, f32 b) { return tensor_apply_biop_scalar(a, b, &__add, ADD); }
tensor *tensor_sub_scalar(tensor *a, f32 b) { return tensor_add_scalar(a, b * -1); }
tensor *tensor_eq_scalar(tensor *a, f32 b) { return tensor_apply_biop_scalar(a, b, &__eq, NEW); }

tensor *tensor_apply_reduceop(tensor *t, i32 axis, bool keepdims, void (*func)(f32*, f32), f32 init, tensor_op op) {
	i32 *nshape;
	if (! keepdims) {
		nshape = (i32*) malloc((t->ndims-1) * sizeof(i32));
		for (i32 i=0; i<axis; ++i) nshape[i]=t->shape[i];
		for (i32 j=axis; j<t->ndims-1; ++j) nshape[j]=t->shape[j+1];
	} else {
		nshape = (i32*) malloc(t->ndims * sizeof(i32));
		for (i32 i=0; i<t->ndims; ++i) nshape[i] = i == axis ? 1 : t->shape[i];
	}

	tensor *r = alloc_tensor(nshape, keepdims ? t->ndims : t->ndims-1, 0, op);

	r->parent_l.type = TENSOR; r->parent_l.value.t = t;
	r->parent_r.type = NONE;
	r->grad_keptdims = keepdims; r->grad_arg = axis;

	i32 *riter = (i32*)malloc(r->ndims * sizeof(i32));
	i32 *iter = (i32*)malloc(t->ndims * sizeof(i32));
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
	free(riter);
	free(iter);

	return r;
}

void __sum(f32 *a, f32 b) { *a += b; }
void __max(f32 *a, f32 b) { *a = *a > b ? *a : b; }
tensor *tensor_sum(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__sum, 0, SUM); }
tensor *tensor_max(tensor *t, i32 axis, bool keepdims) { return tensor_apply_reduceop(t, axis, keepdims, &__max, -INFINITY, MAX); }
tensor *tensor_mean(tensor *t, i32 axis, bool keepdims) {
	tensor *sum = tensor_sum(t, axis, keepdims); sum->free_after_grad = true;
	return tensor_div_scalar(sum, (f32)t->shape[axis]); 
}

tensor *tensor_gemm(tensor *a, tensor *b) {
	if (a->shape[1] != b->shape[0]) return NULL;
	i32 ash[3] = { a->shape[0], a->shape[1], 1 };
	i32 bsh[3] = { 1, b->shape[0], b->shape[1] };

	a = tensor_reshape(a, ash, 3); a->free_after_grad = true;
	b = tensor_reshape(b, bsh, 3); b->free_after_grad = true;
	tensor *c = tensor_mul(a, b); c->free_after_grad = true;

	return tensor_sum(c, 1, false);
}

tensor *tensor_softmax(tensor *t) {
	tensor *max = tensor_max(t, 1, true); max->free_after_grad = true;
	t = tensor_sub(t, max); t->free_after_grad = true;
	t = tensor_exp(t); t->free_after_grad = true;
	tensor *bottom = tensor_sum(t, 1, true); bottom->free_after_grad = true;
	return tensor_div(t, bottom);
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
	t->grad = alloc_tensor(t->shape, t->ndims, 0, NEW);
}

tensor *_unreduce_tensor(tensor *from, tensor *node, tensor *parent) {
	tensor *g = alloc_tensor(node->shape, node->ndims, 0, NEW);
	copy_data(g, from);
	tensor *pg = alloc_tensor(parent->shape, parent->ndims, 0, NEW);
	i32 axis = node->grad_arg;

	if (! node->grad_keptdims) {
		i32 *nshape = (i32*) malloc(parent->ndims * sizeof(i32));
		nshape[axis]=1;
		for (int i=0; i<axis; ++i) nshape[i]=parent->shape[i];
		for (int i=axis+1; i<parent->ndims; ++i) nshape[i]=parent->shape[i];
		g = tensor_reshape(g, nshape, parent->ndims);
		free(nshape);
	}

	i32 *bstrides = broadcast_strides(g);
	i32 *indices = (i32*) malloc(parent->ndims * sizeof(i32));
	memset(indices, 0, parent->ndims * sizeof(i32));
	do {
		f32 *pv = tensor_getitem(pg, parent->strides, indices);
		*pv = *tensor_getitem(g, bstrides, indices);
	} while (inc_shapeindex(indices, parent->shape, parent->ndims) != -1);
	free(indices);
	free(bstrides);
	free(g);
	return pg;
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

	t->grad = alloc_tensor(t->shape, t->ndims, 1.0f, NEW);
	for (i32 i=n-1; i>0; i--) {
		tensor *node = topo[i];

		try_init_parent_grad(&node->parent_r);
		try_init_parent_grad(&node->parent_l);

		tensor *g = (tensor*)node->grad;
		f32 pow; tensor *lp; tensor *lg;
		tensor *contrib; tensor *mask; tensor *max_broadcast;

		if (node->parent_op != NEW) {
			lp = (tensor*)node->parent_l.value.t;
			lg = (tensor*)lp->grad;
			contrib = duplicate_tensor(lg);
		}
		switch (node->parent_op) {
			case NEW: break;
			case RESHAPE: lp->grad = tensor_reshape(g, lp->shape, lp->ndims); break;
			case POW: lp->grad = tensor_mul(g, tensor_mul_scalar(tensor_pow(lp, node->grad_arg-1), node->grad_arg)); break;
			case EXP: lp->grad = tensor_mul(g, node); break;
			case LOG: lp->grad = tensor_mul(g, tensor_pow(lp, -1)); break;
			case ADD:
				if (node->parent_r.type == TENSOR) {
					tensor *rp = (tensor*)node->parent_r.value.t;
					tensor *rg = (tensor*)rp->grad;
					tensor *g_r = _unbroadcast_grad(g, rp);
					tensor *g_l = _unbroadcast_grad(g, lp);
					copy_data(rg, g_r);
					copy_data(lg, g_l);
				} else {
					tensor *g_l = _unbroadcast_grad(g, lp);
					copy_data(lg, g_l);
				}
				break;	
			case MUL:
				if (node->parent_r.type == TENSOR) {
					tensor *rp = (tensor*)node->parent_r.value.t;
					tensor *gl = tensor_mul(g, rp);
					tensor *gr = tensor_mul(g, lp);
					lp->grad = _unbroadcast_grad(gl, lp);
					rp->grad = _unbroadcast_grad(gr, rp);
				} else {
					lp->grad = _unbroadcast_grad(tensor_mul_scalar(g, node->parent_r.value.s), lp);
				}
				break;
			case SUM: lp->grad = _unreduce_tensor(g, node, lp); break;
			case MAX:
				lp->grad = _unreduce_tensor(g, node, lp);
				mask = tensor_eq(_unreduce_tensor(node, node, lp), lp);
				lp->grad = tensor_mul((tensor*)lp->grad, mask);
				break;
			case RELU: lp->grad = tensor_mul(g, _tensor_reluback(lp)); break;
			default: continue; break;
		}
		if (node->parent_op != NEW) lp->grad = tensor_add((tensor*)lp->grad, contrib);
	}

	for (int i=0; i<n; ++i)
		if (topo[i]->free_after_grad)
			free_tensor(topo[i]);
}

tensor *duplicate_tensor(tensor *t) {
	tensor *nt = (tensor*) malloc(sizeof(tensor));
	memcpy(nt, t, sizeof(tensor));

	nt->data = (f32*)malloc(get_size(t->shape, t->ndims) * sizeof(f32));
	nt->shape = (i32*)malloc(t->ndims * sizeof(i32));
	nt->strides = (i32*)malloc(t->ndims * sizeof(i32));
	copy_data(nt, t);
	memcpy(nt->shape, t->shape, t->ndims * sizeof(i32));
	memcpy(nt->strides, t->strides, t->ndims * sizeof(i32));
	return nt;
}

void free_tensor(tensor *t) {
	free(t->data);
	free(t->strides);
	free(t->shape);
	free(t->grad);
	free(t);
}

tensor *alloc_tensor(i32 *shape, i32 ndims, f32 init, tensor_op op) {
	i32 size = get_size(shape, ndims);

	tensor *t = (tensor*) malloc(sizeof(tensor));
	i32 *nshape = (i32*) malloc(ndims * sizeof(i32));
	i32 *strides = (i32*) malloc(ndims * sizeof(i32));
	f32 *data = (f32*) malloc(size * sizeof(f32));

	calculate_strides(shape, ndims, &strides);

	for (int i=0; i<size; ++i) data[i]=init;
	memcpy(nshape, shape, ndims * sizeof(i32));

	t->parent_op = op;
	t->parent_l.type = NONE; t->parent_r.type = NONE;
	t->free_after_grad = false;
	t->data = data;
	t->grad = NULL;
	t->ndims = ndims;
	t->shape = nshape;
	t->strides = strides;

	return t;
}

typedef struct {
	tensor *weights;
	tensor *bias;
} layer_linear;

tensor *linear_forward(layer_linear *layer, tensor *x) {
	tensor *y = tensor_gemm(x, layer->weights);
	return tensor_add(y, layer->bias);
}

typedef struct {
	tensor *kernel;
	tensor *bias;
} layer_conv2d;


tensor *tensor_pad(tensor *t, i32 ph, i32 pw) {
	i32 *nshape = (i32*)malloc(t->ndims * sizeof(i32));

	nshape[0]=t->shape[0];
	nshape[1]=t->shape[1] + ph * 2;
	nshape[2]=t->shape[2] + pw * 2;
	nshape[3]=t->shape[3];

	// TODO: Copy autograd info??
	tensor *pt = alloc_tensor(nshape, t->ndims, 0, NEW);
	int indices[4] = { 0, 0, 0, 0 }; int pindices[4];
	do {
		memcpy(pindices, indices, 4 * sizeof(i32));
		pindices[1]+=ph; pindices[2]+=pw;
		f32 *c = tensor_getitem(pt, pt->strides, pindices);
		*c = *tensor_getitem(t, t->strides, indices);
	} while(inc_shapeindex(indices, t->shape, t->ndims) != -1);
	free(nshape);
	return pt;
}

tensor *tensor_im2col(tensor *im, i32 kh, i32 kw, i32 sh, i32 sw, i32 ph, i32 pw) {
	i32 N = im->shape[0], h = im->shape[1], w=im->shape[2], c = im->shape[3];
	i32 oh = floor((h - kh + ph*2) / sh) + 1, ow = floor((w - kw + pw*2) / sw) + 1;
	im = tensor_pad(im, ph, pw);
	i32 cshape[2] = { N * oh * ow, c * kh * kw};
	tensor *cols = alloc_tensor(cshape, 2,0, NEW);

	// TODO: parallelize?
	i32 row=0;
	// Iterate over # of distinct kernel windows
	for (int n=0; n<N; ++n) {
		for (int j=0; j<oh; ++j) {
			for (int k=0; k<ow; ++k) {
				// Copy current kernel window int col
				i32 col=0;
				for (int kj=0; kj<kh; ++kj) {
					for (int kk=0; kk<kw; ++kk) {
						for (int cc=0; cc<c; ++cc) {
							i32 iter[4] = { n, kj+sh, kk+sw, cc };
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

tensor *conv2d_forward(tensor *x) {
}

typedef struct {
	int *m, *v;
	i32 step_size, nparams;
	f32 b1, b2;
	tensor **params;
} optim_ADAM;
