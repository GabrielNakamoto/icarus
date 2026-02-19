#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float f32;
typedef int i32;

typedef struct {
	int ndims;
	int *shape;
	int *strides;
	f32 *data;
} tensor;

tensor *alloc_tensor(i32 *shape, i32 ndims);
f32 *tensor_getitem(tensor *t, i32 *strides, i32 *shape);

// Reshape Ops
tensor *tensor_reshape(tensor *t, i32 *newshape, i32 newndims);

// Elementwise Ops
tensor *tensor_mul(tensor *a, tensor *b);
tensor *tensor_add(tensor *a, tensor *b);

// Reduce Ops
tensor *tensor_reduce(tensor *t, i32 axis);
tensor *tensor_sum(tensor *t, i32 axis);

// Composed Ops
tensor *tensor_gemm(tensor *a, tensor *b);

void print_shape(tensor *t) {
	printf("(");
	for (int i=0; i<t->ndims; ++i) {
		printf("%d", t->shape[i]);
		if (i<t->ndims-1) printf(",");
	}
	printf(")\n");
}

i32 get_size(i32 *shape, i32 ndims) {
	i32 size = 1;
	for (i32 i=0; i<ndims; ++i) size *= shape[i];
	return size;
}

// Stride helpers
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
	for (int i=0; i<ndims; ++i) bshape[i] = a_sh[i] == 1 ? b_sh[i] : a_sh[i];
	return bshape;
}

int inc_shapeindex(int *indices, int *shape, int ndims) {
	indices[ndims-1]++;
	for (int i=ndims-1; i>=0; i--) {
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

int tensor_getidx(int *strides, int ndims, int *indices) {
	int idx = 0;
	for (int i=0; i<ndims; ++i) idx += strides[i] * indices[i];
	return idx;
}

f32 *tensor_getitem(tensor *t, int *strides, int *indices) {
	return t->data + tensor_getidx(strides, t->ndims, indices);
}


tensor *tensor_reshape(tensor *t, int *newshape, int ndims) {
	if (get_size(newshape, ndims) != get_size(t->shape, t->ndims)) return NULL;

	if (ndims > t->ndims) {
		t->shape = (int*) realloc(t->shape, ndims * sizeof(int));
		t->strides = (int*) realloc(t->strides, ndims * sizeof(int));
	}
	t->ndims = ndims;
	memcpy(t->shape, newshape, ndims * sizeof(int));

	calculate_strides(t->shape, ndims, &t->strides);
	return t;
}

/*
* Binary Elementwise Ops
*/
tensor *tensor_apply_biop(tensor *a, tensor *b, f32 (*op)(f32, f32)) {
	i32 *cshape = broadcast_shape(a->shape, b->shape, a->ndims);
	tensor *c = alloc_tensor(cshape, a->ndims);

	i32 *a_bstrides = broadcast_strides(a); i32 *b_bstrides = broadcast_strides(b);
	i32 *iter = (i32*) malloc(a->ndims * sizeof(i32));
	memset(iter, 0, a->ndims * sizeof(i32));

	do {
		f32 *av = tensor_getitem(a, a_bstrides, iter);
		f32 *bv = tensor_getitem(b, b_bstrides, iter);
		f32 *cv = tensor_getitem(c, c->strides, iter);
		*cv = op(*av, *bv);
	} while (inc_shapeindex(iter, c->shape, c->ndims) != -1);

	free(cshape);
	return c;
}

f32 _mul(f32 a, f32 b) { return a * b; }
tensor *tensor_mul(tensor *a, tensor *b) {
	if (a->ndims != b->ndims) return NULL;
	return tensor_apply_biop(a, b, &_mul);
}

f32 _add(f32 a, f32 b) { return a + b; }
tensor *tensor_add(tensor *a, tensor *b) {
	if (a->ndims != b->ndims) return NULL;
	return tensor_apply_biop(a, b, &_mul);
}


/*
* Reduce Ops
*/
tensor *tensor_apply_reduceop(tensor *t, int axis, void (*op)(f32*, f32)) {
	int *nshape = (int*) malloc((t->ndims-1) * sizeof(int));
	for (int i=0; i<axis; ++i) nshape[i]=t->shape[i];
	for (int j=axis; j<t->ndims-1; ++j) nshape[j]=t->shape[j+1];

	tensor *r = alloc_tensor(nshape, t->ndims-1);

	int *riter = (int*)malloc(r->ndims * sizeof(int));
	int *iter = (int*)malloc(t->ndims * sizeof(int));
	memset(riter, 0, r->ndims * sizeof(int));

	do {
		for (int i=0; i<t->ndims; ++i) iter[i + (i >= axis)]=riter[i];
		f32 *a = tensor_getitem(r, r->strides, riter);
		for (int i=0; i<t->shape[axis]; ++i) {
			iter[axis]=i;
			op(a, *tensor_getitem(t, t->strides, iter));
		}
	} while (inc_shapeindex(riter, r->shape, r->ndims) != -1);
	free(riter);
	free(iter);

	return r;
}

void _redu_add(f32 *a, f32 b) { *a += b; }
tensor *tensor_sum(tensor *t, int axis) {
	return tensor_apply_reduceop(t, axis, &_redu_add);
}

tensor *tensor_gemm(tensor *a, tensor *b) {
	if (a->shape[1] != b->shape[0]) return NULL;

	int ash[3] = { a->shape[0], a->shape[1], 1 };
	int bsh[3] = { 1, b->shape[0], b->shape[1] };

	a = tensor_reshape(a, ash, 3);
	b = tensor_reshape(b, bsh, 3);

	tensor *c = tensor_mul(a, b);

	return tensor_sum(c, 1);
}

tensor *alloc_tensor(i32 *shape, i32 ndims) {
	int size = get_size(shape, ndims);

	tensor *t = (tensor*) malloc(sizeof(tensor));
	i32 *nshape = (i32*) malloc(ndims * sizeof(i32));
	i32 *strides = (i32*) malloc(ndims * sizeof(i32));
	f32 *data = (f32*) malloc(size * sizeof(f32));

	calculate_strides(shape, ndims, &strides);

	memset(data, 0, size * sizeof(f32));
	memcpy(nshape, shape, ndims * sizeof(i32));

	t->data = data;
	t->ndims = ndims;
	t->shape = nshape;
	t->strides = strides;

	return t;
}
