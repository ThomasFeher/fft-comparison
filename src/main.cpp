#include <chrono>
#include <iostream>
#include <vector>

#include <armadillo>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

namespace {
	uint32_t num = 1;
	uint32_t fft_size = 48000;
	uint32_t conv_ir_size = 72000;
	//uint32_t conv_ir_size = 10;
	uint32_t conv_sig_size = 5760000;
	//uint32_t conv_sig_size = 48000;
}

duration<double> Convolution(uint32_t ir_size, uint32_t sig_size) {
	arma::fvec sig {arma::randn<arma::fvec>(sig_size+ir_size-1)};
	sig.subvec(sig_size,sig_size+ir_size-2) = arma::zeros<arma::fvec>(ir_size-1);
	arma::fvec ir {arma::randn<arma::fvec>(ir_size)};
	arma::fvec output (sig_size+ir_size-1);

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		for (uint32_t sample_cnt=0;sample_cnt<sig_size;++sample_cnt) {
			for (uint32_t ir_cnt=0;ir_cnt<ir_size;++ir_cnt) {
				output[sample_cnt] += sig[sample_cnt+ir_cnt] * ir[ir_cnt];
			}
		}
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(output);
	auto end = high_resolution_clock::now();

	return end - begin;
}

duration<double> ArmadilloConv(uint32_t ir_size, uint32_t sig_size) {
	arma::fvec sig {arma::randn<arma::fvec>(sig_size+ir_size-1)};
	sig.subvec(sig_size,sig_size+ir_size-2) = arma::zeros<arma::fvec>(ir_size-1);
	arma::fvec ir {arma::randn<arma::fvec>(ir_size)};
	arma::fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::conv(sig, ir);
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(output);
	auto end = high_resolution_clock::now();

	return end - begin;
}

duration<double> ArmadilloFftConv(uint32_t ir_size, uint32_t sig_size) {
	arma::fvec sig {arma::randn<arma::fvec>(sig_size+ir_size-1)};
	sig.subvec(sig_size,sig_size+ir_size-2) = arma::zeros<arma::fvec>(ir_size-1);
	arma::fvec ir {arma::randn<arma::fvec>(ir_size)};
	arma::cx_fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(arma::fft(sig) % arma::fft(ir,sig_size+ir_size-1));
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return end - begin;
}

duration<double> ArmadilloFftPow2Conv(uint32_t ir_size, uint32_t sig_size) {
	uint32_t size = pow(2,ceil(log2(sig_size+ir_size-1)));
	arma::fvec sig {arma::randn<arma::fvec>(sig_size)};
	arma::fvec ir {arma::randn<arma::fvec>(ir_size)};
	arma::cx_fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(arma::fft(sig,size) % arma::fft(ir,size));
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return end - begin;
}

duration<double> ArmadilloFft(uint32_t fft_size) {
	arma::fvec input {arma::randn<arma::fvec>(fft_size)};
	arma::cx_fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::fft(input);
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return end - begin;
}

duration<double> ArmadilloIFft(uint32_t fft_size) {
	arma::cx_fvec input {arma::randn<arma::cx_fvec>(fft_size)};
	arma::cx_fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(input);
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return end - begin;
}

int main(int argc, char* argv[]) {
	auto arma_fft_time = ArmadilloFft(::fft_size);
	std::cout << "Armadillo FFT: " << arma_fft_time.count() << std::endl;

	auto arma_ifft_time = ArmadilloIFft(::fft_size);
	std::cout << "Armadillo iFFT: " << arma_ifft_time.count() << std::endl;

	auto arma_fft_pow2_conv_time = ArmadilloFftPow2Conv(::conv_ir_size, ::conv_sig_size);
	std::cout << "Armadillo FFT-Pow2-convolution: " << arma_fft_pow2_conv_time.count() << std::endl;

	auto arma_fft_conv_time = ArmadilloFftConv(::conv_ir_size, ::conv_sig_size);
	std::cout << "Armadillo FFT-convolution: " << arma_fft_conv_time.count() << std::endl;

	auto arma_conv_time = ArmadilloConv(::conv_ir_size, ::conv_sig_size);
	std::cout << "Armadillo convolution: " << arma_conv_time.count() << std::endl;

	auto conv_time = Convolution(::conv_ir_size, ::conv_sig_size);
	std::cout << "convolution: " << conv_time.count() << std::endl;
}
