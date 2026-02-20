#include "icarus.h"
#include <assert.h>

#define EPS 1e-3f
#define GRAD_EPS 5e-2f

static int tests_passed = 0;
static int tests_failed = 0;

static int check_close(f32 a, f32 b, f32 tol) {
	f32 diff = fabsf(a - b);
	f32 scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
	return diff / scale < tol;
}

static void check_grad(tensor *t, f32 *expected, int size, const char *name) {
	tensor *g = (tensor*)t->grad;
	if (!g) {
		printf("  FAIL %s: grad is NULL\n", name);
		tests_failed++;
		return;
	}
	int ok = 1;
	for (int i = 0; i < size; i++) {
		if (!check_close(g->data[i], expected[i], GRAD_EPS)) {
			printf("  FAIL %s: grad[%d]=%.6f expected=%.6f\n",
				   name, i, g->data[i], expected[i]);
			ok = 0;
		}
	}
	if (ok) { printf("  PASS %s\n", name); tests_passed++; }
	else tests_failed++;
}

// Compute numerical gradient by finite differences on a fresh copy each time
// fn takes a single tensor, returns a tensor whose elements are summed for a scalar loss
typedef tensor* (*unary_fn)(tensor *);

static void numerical_grad_unary(f32 *input_data, i32 *shape, i32 ndims, unary_fn fn, f32 *out_grad) {
	int size = get_size(shape, ndims);
	f32 h = EPS;

	for (int i = 0; i < size; i++) {
		// f(x + h)
		tensor *tp = alloc_tensor(shape, ndims, 0, NEW);
		memcpy(tp->data, input_data, size * sizeof(f32));
		tp->data[i] += h;
		tensor *outp = fn(tp);
		int out_size = get_size(outp->shape, outp->ndims);
		f32 fplus = 0;
		for (int j = 0; j < out_size; j++) fplus += outp->data[j];

		// f(x - h)
		tensor *tm = alloc_tensor(shape, ndims, 0, NEW);
		memcpy(tm->data, input_data, size * sizeof(f32));
		tm->data[i] -= h;
		tensor *outm = fn(tm);
		out_size = get_size(outm->shape, outm->ndims);
		f32 fminus = 0;
		for (int j = 0; j < out_size; j++) fminus += outm->data[j];

		out_grad[i] = (fplus - fminus) / (2.0f * h);
	}
}

static void check_numerical(f32 *input_data, i32 *shape, i32 ndims,
                             unary_fn fn, tensor *t, const char *name) {
	int size = get_size(shape, ndims);
	f32 num[64];
	numerical_grad_unary(input_data, shape, ndims, fn, num);
	tensor *g = (tensor*)t->grad;
	if (!g) {
		printf("  FAIL %s: grad is NULL\n", name);
		tests_failed++;
		return;
	}
	int ok = 1;
	for (int i = 0; i < size; i++) {
		if (!check_close(g->data[i], num[i], GRAD_EPS)) {
			printf("  FAIL %s: grad[%d] analytical=%.6f numerical=%.6f\n",
				   name, i, g->data[i], num[i]);
			ok = 0;
		}
	}
	if (ok) { printf("  PASS %s\n", name); tests_passed++; }
	else tests_failed++;
}

// Helper
static tensor *make_tensor(i32 *shape, i32 ndims, f32 *data) {
	tensor *t = alloc_tensor(shape, ndims, 0, NEW);
	memcpy(t->data, data, get_size(shape, ndims) * sizeof(f32));
	return t;
}

// ============ Wrapper functions for numerical gradient ============
static tensor *fn_pow2(tensor *t) { return tensor_pow(t, 2.0f); }
static tensor *fn_pow3(tensor *t) { return tensor_pow(t, 3.0f); }
static tensor *fn_exp(tensor *t) { return tensor_exp(t); }
static tensor *fn_log(tensor *t) { return tensor_log(t); }
static tensor *fn_sqrt(tensor *t) { return tensor_sqrt(t); }
static tensor *fn_relu(tensor *t) { return tensor_relu(t); }
static tensor *fn_mul_scalar(tensor *t) { return tensor_mul_scalar(t, 3.0f); }
static tensor *fn_add_scalar(tensor *t) { return tensor_add_scalar(t, 5.0f); }
static tensor *fn_sum_axis0(tensor *t) { return tensor_sum(t, 0, false); }
static tensor *fn_sum_axis1(tensor *t) { return tensor_sum(t, 1, false); }
static tensor *fn_sum_axis0_kd(tensor *t) { return tensor_sum(t, 0, true); }
static tensor *fn_max_axis1(tensor *t) { return tensor_max(t, 1, false); }
static tensor *fn_mean_axis1(tensor *t) { return tensor_mean(t, 1, false); }
static tensor *fn_softmax(tensor *t) { return tensor_softmax(t); }
static tensor *fn_chain_exp_pow(tensor *t) { return tensor_exp(tensor_pow(t, 2.0f)); }
static tensor *fn_softplus(tensor *t) { return tensor_log(tensor_add_scalar(tensor_exp(t), 1.0f)); }
static tensor *fn_chain_sum_relu(tensor *t) {
	tensor *r = tensor_relu(t);
	r = tensor_sum(r, 1, false);
	return r;
}

// ============ Tests ============

void test_pow() {
	printf("test_pow:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {1, 2, 3, 4, 5, 6};

	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_pow2(t));
	// d/dx x^2 = 2x
	f32 expected[] = {2, 4, 6, 8, 10, 12};
	check_grad(t, expected, 6, "pow2");

	t = make_tensor(sh, 2, d);
	tensor_backward(fn_pow3(t));
	// d/dx x^3 = 3x^2
	f32 expected3[] = {3, 12, 27, 48, 75, 108};
	check_grad(t, expected3, 6, "pow3");
}

void test_exp() {
	printf("test_exp:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {0.5, 1.0, -0.5, 0.0};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_exp(t));
	// d/dx e^x = e^x
	f32 expected[] = {expf(0.5), expf(1.0), expf(-0.5), expf(0.0)};
	check_grad(t, expected, 4, "exp");
}

void test_log() {
	printf("test_log:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {1.0, 2.0, 3.0, 0.5};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_log(t));
	// d/dx ln(x) = 1/x
	f32 expected[] = {1.0, 0.5, 1.0/3.0, 2.0};
	check_grad(t, expected, 4, "log");
}

void test_sqrt() {
	printf("test_sqrt:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {1.0, 4.0, 9.0, 16.0};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_sqrt(t));
	// d/dx x^0.5 = 0.5 * x^-0.5
	f32 expected[] = {0.5, 0.25, 1.0/6.0, 0.125};
	check_grad(t, expected, 4, "sqrt");
}

void test_relu() {
	printf("test_relu:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {-2, 0.5, -1, 3, 0.1, -0.1};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_relu(t));
	// d/dx relu(x) = 1 if x > 0, else 0
	f32 expected[] = {0, 1, 0, 1, 1, 0};
	check_grad(t, expected, 6, "relu");
}

void test_mul_scalar() {
	printf("test_mul_scalar:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {1, 2, 3, 4};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_mul_scalar(t));
	f32 expected[] = {3, 3, 3, 3};
	check_grad(t, expected, 4, "mul_scalar(3)");
}

void test_add_scalar() {
	printf("test_add_scalar:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {1, 2, 3, 4};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_add_scalar(t));
	f32 expected[] = {1, 1, 1, 1};
	check_grad(t, expected, 4, "add_scalar(5)");
}

void test_add_tensors() {
	printf("test_add_tensors:\n");
	i32 sh[] = {2, 2};
	f32 da[] = {1, 2, 3, 4}, db[] = {5, 6, 7, 8};
	tensor *a = make_tensor(sh, 2, da);
	tensor *b = make_tensor(sh, 2, db);
	tensor_backward(tensor_add(a, b));
	f32 ones[] = {1, 1, 1, 1};
	check_grad(a, ones, 4, "add_grad_a");
	check_grad(b, ones, 4, "add_grad_b");
}

void test_mul_tensors() {
	printf("test_mul_tensors:\n");
	i32 sh[] = {2, 2};
	f32 da[] = {1, 2, 3, 4}, db[] = {5, 6, 7, 8};
	tensor *a = make_tensor(sh, 2, da);
	tensor *b = make_tensor(sh, 2, db);
	tensor_backward(tensor_mul(a, b));
	// d(a*b)/da = b, d(a*b)/db = a
	check_grad(a, db, 4, "mul_grad_a");
	check_grad(b, da, 4, "mul_grad_b");
}

void test_sub_tensors() {
	printf("test_sub_tensors:\n");
	i32 sh[] = {2, 2};
	f32 da[] = {5, 6, 7, 8}, db[] = {1, 2, 3, 4};
	tensor *a = make_tensor(sh, 2, da);
	tensor *b = make_tensor(sh, 2, db);
	tensor_backward(tensor_sub(a, b));
	f32 ones[] = {1, 1, 1, 1};
	f32 neg_ones[] = {-1, -1, -1, -1};
	check_grad(a, ones, 4, "sub_grad_a");
	check_grad(b, neg_ones, 4, "sub_grad_b");
}

void test_div_tensors() {
	printf("test_div_tensors:\n");
	i32 sh[] = {2, 2};
	f32 da[] = {6, 8, 10, 12}, db[] = {2, 4, 5, 3};
	tensor *a = make_tensor(sh, 2, da);
	tensor *b = make_tensor(sh, 2, db);
	tensor_backward(tensor_div(a, b));
	// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
	f32 ega[4], egb[4];
	for (int i = 0; i < 4; i++) {
		ega[i] = 1.0f / db[i];
		egb[i] = -da[i] / (db[i] * db[i]);
	}
	check_grad(a, ega, 4, "div_grad_a");
	check_grad(b, egb, 4, "div_grad_b");
}

void test_sum_reduce() {
	printf("test_sum_reduce:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {1, 2, 3, 4, 5, 6};

	// sum axis 1: grad should be all 1s
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_sum_axis1(t));
	f32 ones[] = {1, 1, 1, 1, 1, 1};
	check_grad(t, ones, 6, "sum_axis1");

	// sum axis 0: grad should be all 1s
	t = make_tensor(sh, 2, d);
	tensor_backward(fn_sum_axis0(t));
	check_grad(t, ones, 6, "sum_axis0");

	// sum axis 0 keepdims
	t = make_tensor(sh, 2, d);
	tensor_backward(fn_sum_axis0_kd(t));
	check_grad(t, ones, 6, "sum_axis0_keepdims");
}

void test_max_reduce() {
	printf("test_max_reduce:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {1, 5, 3, 6, 2, 4};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_max_axis1(t));
	// Only the max element per row gets gradient 1, rest 0
	// Row 0: max=5 (idx 1), Row 1: max=6 (idx 0)
	f32 expected[] = {0, 1, 0, 1, 0, 0};
	check_grad(t, expected, 6, "max_axis1");
}

void test_mean_reduce() {
	printf("test_mean_reduce:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {1, 2, 3, 4, 5, 6};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_mean_axis1(t));
	// mean = sum / N, so grad = 1/N for each element
	f32 expected[] = {1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3};
	check_grad(t, expected, 6, "mean_axis1");
}

void test_gemm() {
	printf("test_gemm:\n");
	i32 sha[] = {2, 3}, shb[] = {3, 2};
	f32 da[] = {1, 2, 3, 4, 5, 6};
	f32 db[] = {7, 8, 9, 10, 11, 12};
	tensor *a = make_tensor(sha, 2, da);
	tensor *b = make_tensor(shb, 2, db);

	tensor *out = tensor_gemm(a, b);

	// Verify forward: C = A @ B
	f32 expected_c[] = {
		1*7+2*9+3*11,  1*8+2*10+3*12,
		4*7+5*9+6*11,  4*8+5*10+6*12
	};
	int ok = 1;
	for (int i = 0; i < 4; i++) {
		if (!check_close(out->data[i], expected_c[i], GRAD_EPS)) {
			printf("  FAIL gemm_forward: out[%d]=%.4f expected=%.4f\n",
				   i, out->data[i], expected_c[i]);
			ok = 0;
		}
	}
	if (ok) { printf("  PASS gemm_forward\n"); tests_passed++; }
	else tests_failed++;

	tensor_backward(out);

	// dL/dA = 1 @ B^T  (upstream grad is all-ones since backward starts with 1s)
	// dL/dA[i,j] = sum_k dL/dC[i,k] * B[j,k] = sum_k B[j,k]
	f32 ega[] = {7+8, 9+10, 11+12, 7+8, 9+10, 11+12};
	// dL/dB = A^T @ 1
	// dL/dB[j,k] = sum_i A[i,j] * dL/dC[i,k] = sum_i A[i,j]
	f32 egb[] = {1+4, 1+4, 2+5, 2+5, 3+6, 3+6};
	check_grad(a, ega, 6, "gemm_grad_a");
	check_grad(b, egb, 6, "gemm_grad_b");
}

void test_softmax() {
	printf("test_softmax:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {1.0, 2.0, 3.0, 1.0, 1.0, 1.0};
	tensor *t = make_tensor(sh, 2, d);

	tensor *out = fn_softmax(t);

	// Rows should sum to 1
	f32 row0 = out->data[0] + out->data[1] + out->data[2];
	f32 row1 = out->data[3] + out->data[4] + out->data[5];
	if (check_close(row0, 1.0f, GRAD_EPS) && check_close(row1, 1.0f, GRAD_EPS)) {
		printf("  PASS softmax_forward\n"); tests_passed++;
	} else {
		printf("  FAIL softmax_forward: row sums %.4f, %.4f\n", row0, row1);
		tests_failed++;
	}

	// Numerical gradient check
	tensor_backward(out);
	check_numerical(d, sh, 2, fn_softmax, t, "softmax_grad");
}

void test_chain_exp_pow() {
	printf("test_chain_exp_pow:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {0.5, 1.0, -0.5, 0.3};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_chain_exp_pow(t));
	// d/dx exp(x^2) = 2x * exp(x^2)
	f32 expected[4];
	for (int i = 0; i < 4; i++) expected[i] = 2 * d[i] * expf(d[i] * d[i]);
	check_grad(t, expected, 4, "chain_exp_pow");
}

void test_softplus() {
	printf("test_softplus:\n");
	i32 sh[] = {2, 2};
	f32 d[] = {-1, 0, 1, 2};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_softplus(t));
	// d/dx log(exp(x) + 1) = exp(x) / (exp(x) + 1) = sigmoid(x)
	f32 expected[4];
	for (int i = 0; i < 4; i++) expected[i] = expf(d[i]) / (expf(d[i]) + 1.0f);
	check_grad(t, expected, 4, "softplus");
}

void test_chain_sum_relu() {
	printf("test_chain_sum_relu:\n");
	i32 sh[] = {2, 3};
	f32 d[] = {-2, 1, -0.5, 3, -1, 0.5};
	tensor *t = make_tensor(sh, 2, d);
	tensor_backward(fn_chain_sum_relu(t));
	// sum(relu(x)): grad = 1 if x > 0, 0 otherwise
	f32 expected[] = {0, 1, 0, 1, 0, 1};
	check_grad(t, expected, 6, "chain_sum_relu");
}

int main() {
	printf("=== Icarus Autograd Tests ===\n\n");

	test_pow();
	test_exp();
	test_log();
	test_sqrt();
	test_relu();

	test_mul_scalar();
	test_add_scalar();

	test_add_tensors();
	test_mul_tensors();
	test_sub_tensors();
	test_div_tensors();

	test_sum_reduce();
	test_max_reduce();
	test_mean_reduce();

	test_gemm();
	test_softmax();

	test_chain_exp_pow();
	test_softplus();
	test_chain_sum_relu();

	printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
	return tests_failed > 0 ? 1 : 0;
}
