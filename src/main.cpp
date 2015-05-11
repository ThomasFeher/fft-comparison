#include <chrono>
#include <iostream>
#include <vector>
#include <utility> // std::pair

#include <armadillo>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

namespace {
	uint32_t num = 1;
	uint32_t fft_size = 48000;
	uint32_t conv_ir_size = 72000;
	uint32_t conv_sig_size = 5760000;
}

std::pair<duration<double>, arma::fvec> Convolution(arma::fvec sig, arma::fvec ir) {
	size_t size = (::conv_sig_size+::conv_ir_size-1);
	ir = arma::flipud(ir);
	arma::fvec sig_new = arma::zeros<arma::fvec>(::conv_sig_size + 2*(::conv_ir_size-1));
	sig_new.subvec(::conv_ir_size - 1, ::conv_sig_size + ::conv_ir_size -2) = sig;
	arma::fvec output (::conv_sig_size + ::conv_ir_size - 1, arma::fill::zeros);

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		for (uint32_t sample_cnt=0;sample_cnt<size;++sample_cnt) {
			for (uint32_t ir_cnt=0;ir_cnt<::conv_ir_size;++ir_cnt) {
				output[sample_cnt] += sig_new[sample_cnt+ir_cnt] * ir[ir_cnt];
			}
		}
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(output);
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, output);
}

std::pair<duration<double>, arma::fvec> ArmadilloConv(arma::fvec sig, arma::fvec ir) {
	arma::fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::conv(sig, ir);
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(output);
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, output);
}

std::pair<duration<double>, arma::fvec> ArmadilloFftConv(arma::fvec sig, arma::fvec ir) {
	arma::cx_fvec output;
	size_t size = sig.size() + ir.size() - 1;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(arma::fft(sig, size) % arma::fft(ir, size));
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, arma::real(output));
}

std::pair<duration<double>, arma::fvec> ArmadilloFftPow2Conv(arma::fvec sig, arma::fvec ir) {
	uint32_t size = pow(2,ceil(log2(::conv_sig_size + ::conv_ir_size - 1)));
	arma::cx_fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(arma::fft(sig,size) % arma::fft(ir,size));
	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output));
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, arma::real(output.subvec(0, ::conv_sig_size + ::conv_ir_size - 2)));
}

std::pair<duration<double>,arma::fvec> ArmadilloFft(arma::fvec input) {
	arma::cx_fvec output_fd;
	arma::cx_fvec output_td;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output_fd = arma::fft(input);
		output_td = arma::ifft(output_fd);

	}
	std::vector<float> output_copy = arma::conv_to<std::vector<float>>::from(arma::real(output_td));
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin,arma::real(output_td));
}

int main(int argc, char* argv[]) {

	// generate input signals
	arma::fvec sig {arma::randn<arma::fvec>(::conv_sig_size)};
	arma::fvec ir {arma::randn<arma::fvec>(::conv_ir_size)};

	auto result_arma_fft = ArmadilloFft(sig);
	std::cout << "Armadillo FFT: " << result_arma_fft.first.count() << std::endl;

	// normal convolution, this is our reference output
	auto result_conv = Convolution(sig, ir);
	std::cout << "convolution: " << result_conv.first.count()
			  << std::endl;

	auto result_arma_fft_pow2_conv = ArmadilloFftPow2Conv(sig, ir);
	std::cout << "Armadillo FFT-Pow2-convolution: "
		      << result_arma_fft_pow2_conv.first.count()
			  << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_fft_pow2_conv.second).max()
			  << std::endl;

	auto result_arma_fft_conv = ArmadilloFftConv(sig, ir);
	std::cout << "Armadillo FFT-convolution: "
		      << result_arma_fft_conv.first.count()
			  << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_fft_conv.second).max()
			  << std::endl;

	auto result_arma_conv = ArmadilloConv(sig, ir);
	std::cout << "Armadillo convolution: "
		      << result_arma_conv.first.count()
			  << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_conv.second).max()
			  << std::endl;
}
