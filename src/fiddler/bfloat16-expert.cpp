
#include <math.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#define N_DIM (4096)
#define M_DIM (14336)

union fi
{
	float f;
	uint32_t u;
};

typedef uint16_t bfloat16;

float u2f(uint32_t u)
{
	fi fi;
	fi.u = u;
	return fi.f;
}

uint32_t f2u(float f)
{
	fi fi;
	fi.f = f;
	return fi.u;
}

bfloat16 float_to_bfloat16(float f)
{
	// ignore denormal and infinity
	uint32_t u = f2u(f);
	uint32_t rounding = 0x7fff + ((u >> 16) & 1);
	u += rounding;
	return bfloat16(u >> 16);
}

float bfloat16_to_float(bfloat16 f)
{
	return u2f(f << 16);
}

float silu(float x)
{
	return x / (1 + exp(-x));
}

static inline void gen_vdpbf16ps_api(float *arg0, bfloat16 *arg1, bfloat16 *arg2)
{
	// Load data from memory into ZMM registers using AVX-512 instructions.
	__m512 zmm0 = _mm512_loadu_ps(arg0); // Assume data at arg0 is in single-precision float format

	// Load 16-bit values from arg1 and arg2 into ZMM registers
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)arg1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)arg2);

	// Perform dot product of pairs of BF16 floating-point values in zmm1 and zmm2,
	// and accumulate the single-precision floating-point result in zmm0.
	zmm0 = _mm512_dpbf16_ps(zmm0, (__m512bh)zmm1, (__m512bh)zmm2);

	// Store the result back to memory.
	_mm512_storeu_ps(arg0, zmm0);
}

void vdpbf16psC(float *dst, const bfloat16 *src1, const bfloat16 *src2)
{
	for (int i = 0; i < N_DIM / 2; i++)
	{
		dst[i] += bfloat16_to_float(src1[i * 2 + 0]) * bfloat16_to_float(src2[i * 2 + 0]);
		dst[i] += bfloat16_to_float(src1[i * 2 + 1]) * bfloat16_to_float(src2[i * 2 + 1]);
	}
}

void vcvtne2ps2bf16C(bfloat16 *dst, const float *src1, const float *src2)
{
	for (int j = 0; j < N_DIM / 32; j++)
	{
		for (int i = 0; i < 16; i++)
		{
			dst[32 * j + i] = float_to_bfloat16(src1[16 * j + i]);
			dst[32 * j + i + 16] = float_to_bfloat16(src2[16 * j + i]);
		}
	}
}

char diff(float x, float y)
{
	return fabs(x - y) < 1e-5f ? 'o' : 'x';
}

// bfloat16 inp[N_TEST][N_DIM];
// float out[N_TEST][N_DIM];

// bfloat16 w1[N_TEST][M_DIM][N_DIM];
// bfloat16 w2[N_TEST][N_DIM][M_DIM];
// bfloat16 w3[N_TEST][M_DIM][N_DIM];

void convert_inp(std::vector<float> &inp, std::vector<bfloat16> &inp_bf)
{
	for (int i = 0; i < N_DIM; i++)
	{
		inp_bf[i] = float_to_bfloat16(inp[i]);
	}
}

// class CPUExpert
// {
// public:
// 	std::vector<std::vector<bfloat16>> w1;
// 	std::vector<std::vector<bfloat16>> w2;
// 	std::vector<std::vector<bfloat16>> w3;
// 	CPUExpert(std::vector<std::vector<float>> &w1_in, std::vector<std::vector<float>> &w2_in, std::vector<std::vector<float>> &w3_in)
// 	{
// 		// print the size of elements in w1
// 		w1 = std::vector<std::vector<bfloat16>>(M_DIM, std::vector<bfloat16>(N_DIM));
// 		w2 = std::vector<std::vector<bfloat16>>(N_DIM, std::vector<bfloat16>(M_DIM));
// 		w3 = std::vector<std::vector<bfloat16>>(M_DIM, std::vector<bfloat16>(N_DIM));
// 		omp_set_num_threads(56);
// #pragma omp parallel for
// 		for (int j = 0; j < M_DIM; j++)
// 		{
// 			for (int i = 0; i < N_DIM; i++)
// 			{
// 				w1[j][i] = float_to_bfloat16(w1_in[j][i]);
// 				w3[j][i] = float_to_bfloat16(w3_in[j][i]);
// 			}
// 		}
// #pragma omp parallel for
// 		for (int j = 0; j < N_DIM; j++)
// 		{
// 			for (int i = 0; i < M_DIM; i++)
// 			{
// 				w2[j][i] = float_to_bfloat16(w2_in[j][i]);
// 			}
// 		}
// 	}
// 	std::vector<float> operator()(std::vector<float> &input)
// 	{
// 		std::vector<bfloat16> inp(N_DIM);
// 		convert_inp(input, inp);
// 		std::vector<float> out(N_DIM);
// 		int num_threads = 56; // replace with your desired number of threads
// 		omp_set_num_threads(num_threads);

// 		bfloat16 in2[M_DIM];

// 		// Get the start time
// 		auto start = std::chrono::high_resolution_clock::now();

// #pragma omp parallel for
// 		for (int j = 0; j < M_DIM; j++)
// 		{
// 			float dst1[N_DIM / 2] = {};
// 			float dst3[N_DIM / 2] = {};
// 			float sum1 = 0;
// 			float sum3 = 0;
// 			for (int i = 0; i < N_DIM / 32; i += 4)
// 			{
// 				// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
// 				gen_vdpbf16ps_api(&dst1[(i + 0) * 16], &inp[(i + 0) * 32], &w1[j][(i + 0) * 32]);
// 				gen_vdpbf16ps_api(&dst1[(i + 1) * 16], &inp[(i + 1) * 32], &w1[j][(i + 1) * 32]);
// 				gen_vdpbf16ps_api(&dst1[(i + 2) * 16], &inp[(i + 2) * 32], &w1[j][(i + 2) * 32]);
// 				gen_vdpbf16ps_api(&dst1[(i + 3) * 16], &inp[(i + 3) * 32], &w1[j][(i + 3) * 32]);

// 				sum1 += dst1[(i + 0) * 16] + dst1[(i + 1) * 16] + dst1[(i + 2) * 16] + dst1[(i + 3) * 16];
// 			}

// 			for (int i = 0; i < N_DIM / 32; i += 4)
// 			{
// 				// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
// 				gen_vdpbf16ps_api(&dst3[(i + 0) * 16], &inp[(i + 0) * 32], &w3[j][(i + 0) * 32]);
// 				gen_vdpbf16ps_api(&dst3[(i + 1) * 16], &inp[(i + 1) * 32], &w3[j][(i + 1) * 32]);
// 				gen_vdpbf16ps_api(&dst3[(i + 2) * 16], &inp[(i + 2) * 32], &w3[j][(i + 2) * 32]);
// 				gen_vdpbf16ps_api(&dst3[(i + 3) * 16], &inp[(i + 3) * 32], &w3[j][(i + 3) * 32]);

// 				sum3 += dst3[(i + 0) * 16] + dst3[(i + 1) * 16] + dst3[(i + 2) * 16] + dst3[(i + 3) * 16];
// 			}

// 			in2[j] = float_to_bfloat16(silu(sum1) * sum3);
// 		}

// #pragma omp parallel for
// 		for (int j = 0; j < N_DIM; j++)
// 		{
// 			float sum2 = 0;
// 			float dst2[M_DIM / 2] = {};

// 			for (int i = 0; i < M_DIM / 32; i += 4)
// 			{
// 				gen_vdpbf16ps_api(&dst2[(i + 0) * 16], &in2[(i + 0) * 32], &w2[j][(i + 0) * 32]);
// 				gen_vdpbf16ps_api(&dst2[(i + 1) * 16], &in2[(i + 1) * 32], &w2[j][(i + 1) * 32]);
// 				gen_vdpbf16ps_api(&dst2[(i + 2) * 16], &in2[(i + 2) * 32], &w2[j][(i + 2) * 32]);
// 				gen_vdpbf16ps_api(&dst2[(i + 3) * 16], &in2[(i + 3) * 32], &w2[j][(i + 3) * 32]);

// 				sum2 += dst2[(i + 0) * 16] + dst2[(i + 1) * 16] + dst2[(i + 2) * 16] + dst2[(i + 3) * 16];
// 			}
// 			out[j] = sum2;
// 		}

// 		// Get the end time
// 		auto end = std::chrono::high_resolution_clock::now();

// 		float print_out = 0;
// 		for (int j = 0; j < N_DIM; j++)
// 		{
// 			print_out += out[j];
// 		}

// 		// Calculate and print the duration
// 		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
// 		std::cout << "Time taken by loop: " << duration << " microseconds " << print_out << std::endl;
// 		return out;
// 	}
// };

class cpu_expert
{
public:
	torch::Tensor w1;
	torch::Tensor w2;
	torch::Tensor w3;
	cpu_expert(torch::Tensor &w1, torch::Tensor &w2, torch::Tensor &w3) : w1(w1), w2(w2), w3(w3) {}

	torch::Tensor operator()(torch::Tensor &inp_tensor)
	{
		int num_threads = 16; // replace with your desired number of threads
		omp_set_num_threads(num_threads);

		bfloat16 in2[M_DIM];

		// Get the start time
		std::cout << "Start running expert" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		bfloat16 *inp = static_cast<bfloat16 *>(inp_tensor.data_ptr());
		float *out = new float[N_DIM];

		auto w1 = static_cast<bfloat16 *>(this->w1.data_ptr());
		auto w2 = static_cast<bfloat16 *>(this->w2.data_ptr());
		auto w3 = static_cast<bfloat16 *>(this->w3.data_ptr());

#pragma omp parallel for
		for (int j = 0; j < M_DIM; j++)
		{
			float dst1[N_DIM / 2] = {};
			float dst3[N_DIM / 2] = {};
			float sum1 = 0;
			float sum3 = 0;
			for (int i = 0; i < N_DIM / 32; i += 4)
			{
				// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
				gen_vdpbf16ps_api(&dst1[(i + 0) * 16], &inp[(i + 0) * 32], &w1[j * N_DIM + (i + 0) * 32]);
				gen_vdpbf16ps_api(&dst1[(i + 1) * 16], &inp[(i + 1) * 32], &w1[j * N_DIM + (i + 1) * 32]);
				gen_vdpbf16ps_api(&dst1[(i + 2) * 16], &inp[(i + 2) * 32], &w1[j * N_DIM + (i + 2) * 32]);
				gen_vdpbf16ps_api(&dst1[(i + 3) * 16], &inp[(i + 3) * 32], &w1[j * N_DIM + (i + 3) * 32]);

				sum1 += dst1[(i + 0) * 16] + dst1[(i + 1) * 16] + dst1[(i + 2) * 16] + dst1[(i + 3) * 16];
			}
			for (int i = 0; i < N_DIM / 32; i += 4)
			{
				// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
				gen_vdpbf16ps_api(&dst3[(i + 0) * 16], &inp[(i + 0) * 32], &w3[j * N_DIM + (i + 0) * 32]);
				gen_vdpbf16ps_api(&dst3[(i + 1) * 16], &inp[(i + 1) * 32], &w3[j * N_DIM + (i + 1) * 32]);
				gen_vdpbf16ps_api(&dst3[(i + 2) * 16], &inp[(i + 2) * 32], &w3[j * N_DIM + (i + 2) * 32]);
				gen_vdpbf16ps_api(&dst3[(i + 3) * 16], &inp[(i + 3) * 32], &w3[j * N_DIM + (i + 3) * 32]);

				sum3 += dst3[(i + 0) * 16] + dst3[(i + 1) * 16] + dst3[(i + 2) * 16] + dst3[(i + 3) * 16];
			}
			in2[j] = float_to_bfloat16(silu(sum1) * sum3);
		}

#pragma omp parallel for
		for (int j = 0; j < N_DIM; j++)
		{
			float sum2 = 0;
			float dst2[M_DIM / 2] = {};

			for (int i = 0; i < M_DIM / 32; i += 4)
			{
				gen_vdpbf16ps_api(&dst2[(i + 0) * 16], &in2[(i + 0) * 32], &w2[j * M_DIM + (i + 0) * 32]);
				gen_vdpbf16ps_api(&dst2[(i + 1) * 16], &in2[(i + 1) * 32], &w2[j * M_DIM + (i + 1) * 32]);
				gen_vdpbf16ps_api(&dst2[(i + 2) * 16], &in2[(i + 2) * 32], &w2[j * M_DIM + (i + 2) * 32]);
				gen_vdpbf16ps_api(&dst2[(i + 3) * 16], &in2[(i + 3) * 32], &w2[j * M_DIM + (i + 3) * 32]);

				sum2 += dst2[(i + 0) * 16] + dst2[(i + 1) * 16] + dst2[(i + 2) * 16] + dst2[(i + 3) * 16];
			}
			out[j] = sum2;
		}
		// Get the end time
		auto end = std::chrono::high_resolution_clock::now();
		float print_out = 0;
		for (int j = 0; j < N_DIM; j++)
		{
			print_out += out[j];
		}

		// Calculate and print the duration
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << "Time taken by loop: " << duration << " microseconds " << print_out << std::endl;
		// create an empty tensor N_DIM long and type bfloat16
		torch::Tensor out_tensor = torch::empty({N_DIM}, torch::kBFloat16);
		auto out_tensor_data = static_cast<bfloat16 *>(out_tensor.data_ptr());
		// convert the output to bfloat16 and copy it to the tensor
		for (int i = 0; i < N_DIM; i++)
		{
			out_tensor_data[i] = float_to_bfloat16(out[i]);
		}
		delete[] out;
		return out_tensor;
	}
};

void expert(bfloat16 *inp, float *out, bfloat16 *w1, bfloat16 *w2, bfloat16 *w3)
{
	int num_threads = 56; // replace with your desired number of threads
	omp_set_num_threads(num_threads);

	bfloat16 in2[M_DIM];

	// Get the start time
	auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
	for (int j = 0; j < M_DIM; j++)
	{
		float dst1[N_DIM / 2] = {};
		float dst3[N_DIM / 2] = {};
		float sum1 = 0;
		float sum3 = 0;
		for (int i = 0; i < N_DIM / 32; i += 4)
		{
			// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
			gen_vdpbf16ps_api(&dst1[(i + 0) * 16], &inp[(i + 0) * 32], &w1[j * N_DIM + (i + 0) * 32]);
			gen_vdpbf16ps_api(&dst1[(i + 1) * 16], &inp[(i + 1) * 32], &w1[j * N_DIM + (i + 1) * 32]);
			gen_vdpbf16ps_api(&dst1[(i + 2) * 16], &inp[(i + 2) * 32], &w1[j * N_DIM + (i + 2) * 32]);
			gen_vdpbf16ps_api(&dst1[(i + 3) * 16], &inp[(i + 3) * 32], &w1[j * N_DIM + (i + 3) * 32]);

			sum1 += dst1[(i + 0) * 16] + dst1[(i + 1) * 16] + dst1[(i + 2) * 16] + dst1[(i + 3) * 16];
		}

		for (int i = 0; i < N_DIM / 32; i += 4)
		{
			// vdpbf16ps(&dst1[i * 16], &src1[i * 32], &src2[j][i * 32]);
			gen_vdpbf16ps_api(&dst3[(i + 0) * 16], &inp[(i + 0) * 32], &w3[j * N_DIM + (i + 0) * 32]);
			gen_vdpbf16ps_api(&dst3[(i + 1) * 16], &inp[(i + 1) * 32], &w3[j * N_DIM + (i + 1) * 32]);
			gen_vdpbf16ps_api(&dst3[(i + 2) * 16], &inp[(i + 2) * 32], &w3[j * N_DIM + (i + 2) * 32]);
			gen_vdpbf16ps_api(&dst3[(i + 3) * 16], &inp[(i + 3) * 32], &w3[j * N_DIM + (i + 3) * 32]);

			sum3 += dst3[(i + 0) * 16] + dst3[(i + 1) * 16] + dst3[(i + 2) * 16] + dst3[(i + 3) * 16];
		}

		in2[j] = float_to_bfloat16(silu(sum1) * sum3);
	}

#pragma omp parallel for
	for (int j = 0; j < N_DIM; j++)
	{
		float sum2 = 0;
		float dst2[M_DIM / 2] = {};

		for (int i = 0; i < M_DIM / 32; i += 4)
		{
			gen_vdpbf16ps_api(&dst2[(i + 0) * 16], &in2[(i + 0) * 32], &w2[j * M_DIM + (i + 0) * 32]);
			gen_vdpbf16ps_api(&dst2[(i + 1) * 16], &in2[(i + 1) * 32], &w2[j * M_DIM + (i + 1) * 32]);
			gen_vdpbf16ps_api(&dst2[(i + 2) * 16], &in2[(i + 2) * 32], &w2[j * M_DIM + (i + 2) * 32]);
			gen_vdpbf16ps_api(&dst2[(i + 3) * 16], &in2[(i + 3) * 32], &w2[j * M_DIM + (i + 3) * 32]);

			sum2 += dst2[(i + 0) * 16] + dst2[(i + 1) * 16] + dst2[(i + 2) * 16] + dst2[(i + 3) * 16];
		}
		out[j] = sum2;
	}

	// Get the end time
	auto end = std::chrono::high_resolution_clock::now();

	float print_out = 0;
	for (int j = 0; j < N_DIM; j++)
	{
		print_out += out[j];
	}

	// Calculate and print the duration
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "Time taken by loop: " << duration << " microseconds " << print_out << std::endl;
}

namespace py = pybind11;
PYBIND11_MODULE(bfloat16_expert, m)
{
	// define class expert above
	// py::class_<cpu_expert>(m, "cpu_expert")
	// 	.def(py::init<torch::Tensor &, torch::Tensor &, torch::Tensor &>())
	// 	.def("__call__", &cpu_expert::operator());
	py::class_<cpu_expert>(m, "cpu_expert")
		.def(py::init<torch::Tensor &, torch::Tensor &, torch::Tensor &>())
		.def("__call__", &cpu_expert::operator());
}

int main()
{

	// allocate memory as vector
	bfloat16 *inp = new bfloat16[N_DIM];
	float *out = new float[N_DIM];

	bfloat16 *w1 = new bfloat16[M_DIM * N_DIM];
	bfloat16 *w2 = new bfloat16[N_DIM * M_DIM];
	bfloat16 *w3 = new bfloat16[M_DIM * N_DIM];

	for (int i = 0; i < N_DIM; i++)
	{
		inp[i] = float_to_bfloat16(i * 0.1f + 1.0f);
	}
	for (int j = 0; j < M_DIM; j++)
	{
		for (int i = 0; i < N_DIM; i++)
		{

			w1[j * N_DIM + i] = float_to_bfloat16(i * 0.3f - 4.0f);
			w3[j * N_DIM + i] = float_to_bfloat16(i * 0.3f - 4.0f);
		}
	}
	for (int j = 0; j < N_DIM; j++)
	{
		for (int i = 0; i < M_DIM; i++)
		{
			w2[j * M_DIM + i] = float_to_bfloat16(i * 0.3f - 4.0f);
		}
	}
	expert(inp, out, w1, w2, w3);
	delete[] inp;
	delete[] out;
	delete[] w1;
	delete[] w2;
	delete[] w3;
}

// g++ -fopenmp -mavx512bf16 -O3 bfloat16-test.cpp