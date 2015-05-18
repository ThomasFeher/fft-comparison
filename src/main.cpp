#include <chrono>
#include <iostream>
#include <vector>
#include <utility> // std::pair
#include <algorithm> // std::max_element()
#include <string> // std::to_string()

#include <armadillo>

#include "AudioFFT/AudioFFT.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using AudioVec = std::vector<float>;
using audiofft::AudioFFT;
using FftwData = std::tuple<AudioFFT& // fft class
                           ,AudioVec& // input vector
						   ,AudioVec& // real vector
						   ,AudioVec& // imag vector
						   ,AudioVec&>; // output vector

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
	AudioVec output_copy = arma::conv_to<AudioVec>::from(output);
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, output);
}

std::pair<duration<double>, arma::fvec> ArmadilloConv(arma::fvec sig, arma::fvec ir) {
	arma::fvec output;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::conv(sig, ir);
	}
	AudioVec output_copy = arma::conv_to<AudioVec>::from(output);
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin, output);
}

duration<double> FftwConv(FftwData sig, FftwData ir) {
	auto sig_in = std::get<1>(sig);
	auto sig_re = std::get<2>(sig);
	auto sig_im = std::get<3>(sig);
	auto ir_in = std::get<1>(ir);
	auto ir_re = std::get<2>(ir);
	auto ir_im = std::get<3>(ir);
	AudioVec re (sig_re.size());
	AudioVec im (sig_re.size());

	auto begin = high_resolution_clock::now();
	std::get<0>(sig).fft(sig_in.data(), sig_re.data(), sig_im.data());
	std::get<0>(ir).fft(ir_in.data(), ir_re.data(), ir_im.data());
	for (size_t cnt=0;cnt<sig_re.size();++cnt) {
		re[cnt] = sig_re[cnt]*ir_re[cnt] - sig_im[cnt]*ir_im[cnt];
		im[cnt] = sig_re[cnt]*ir_im[cnt] + sig_im[cnt]*ir_re[cnt];
	}
	std::get<0>(sig).ifft(std::get<4>(sig).data(), re.data(), im.data());
	auto end = high_resolution_clock::now();

	return end - begin;
}

std::pair<duration<double>, arma::fvec> ArmadilloFftConv(arma::fvec sig, arma::fvec ir) {
	arma::cx_fvec output;
	size_t size = sig.size() + ir.size() - 1;

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		output = arma::ifft(arma::fft(sig, size) % arma::fft(ir, size));
	}
	AudioVec output_copy = arma::conv_to<AudioVec>::from(arma::real(output));
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
	AudioVec output_copy = arma::conv_to<AudioVec>::from(arma::real(output));
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
	AudioVec output_copy = arma::conv_to<AudioVec>::from(arma::real(output_td));
	auto end = high_resolution_clock::now();

	return std::make_pair(end - begin,arma::real(output_td));
}

int main(int argc, char* argv[]) {

	std::cout << "generating input signals" << std::endl;
	arma::fvec sig {arma::randn<arma::fvec>(::conv_sig_size)};
	arma::fvec ir {arma::randn<arma::fvec>(::conv_ir_size)};
	size_t out_size = ::conv_sig_size + ::conv_ir_size - 1;

	std::cout << "initializing FFTW" << std::flush;
	size_t fft_size (pow(2,ceil(log2(out_size))));
	AudioVec sig_vec  (fft_size, 0);
	std::copy_n(sig.begin(), ::conv_sig_size, sig_vec.begin());
	AudioVec ir_vec (fft_size, 0);
	std::copy_n(ir.begin(), ::conv_ir_size, ir_vec.begin());
	AudioVec sig_re(AudioFFT::ComplexSize(fft_size)); 
	AudioVec ir_re(AudioFFT::ComplexSize(fft_size));
	AudioVec sig_im(AudioFFT::ComplexSize(fft_size)); 
	AudioVec ir_im(AudioFFT::ComplexSize(fft_size)); 
	AudioVec output(fft_size);
	AudioFFT fft_sig;
	auto begin = high_resolution_clock::now();
	fft_sig.init(fft_size);
	auto end = high_resolution_clock::now();
	std::cout << "\n\tduration: " << (std::chrono::duration_cast<duration<double>>(end - begin)).count() << std::endl;
	AudioFFT fft_ir;
	fft_ir.init(fft_size);
	FftwData sig_data = std::forward_as_tuple(fft_sig, sig_vec, sig_re, sig_im, output);
	FftwData ir_data = std::forward_as_tuple(fft_ir, ir_vec, ir_re, ir_im, output);

	std::cout << "Armadillo FFT: " << std::flush;
	auto result_arma_fft = ArmadilloFft(sig);
	std::cout << "\n\tduration: " << result_arma_fft.first.count() << std::endl;

	// normal convolution, this is our reference output
	std::cout << "convolution: " << std::flush;
	auto result_conv = Convolution(sig, ir);
	std::cout << "\n\tduration: " << result_conv.first.count() << std::endl;

	std::cout << "Armadillo FFT-Pow2-convolution: " << std::flush;
	auto result_arma_fft_pow2_conv = ArmadilloFftPow2Conv(sig, ir);
	std::cout << "\n\tduration: " << result_arma_fft_pow2_conv.first.count();
	std::cout << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_fft_pow2_conv.second).max()
			  << std::endl;

	std::cout << "FFTW FFT-Pow2-convolution: " << std::flush;
	auto result_fftw_pow2_conv = FftwConv( sig_data, ir_data);
	AudioVec diff;
	std::transform(
			result_conv.second.begin()
			, result_conv.second.end()
			, std::get<4>(sig_data).begin()
			, std::back_inserter(diff)
			, [](float a, float b) { return fabs(a-b); }
			);
	std::cout << "\n\tduration: " << result_fftw_pow2_conv.count();
	std::cout << "\n\tmaximum difference of result: "
			  << std::to_string(*std::max_element(diff.begin(), diff.end()))
			  << std::endl;

	std::cout << "Armadillo FFT-convolution: " << std::flush;
	auto result_arma_fft_conv = ArmadilloFftConv(sig, ir);
	std::cout << "\n\tduration: " << result_arma_fft_conv.first.count();
	std::cout << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_fft_conv.second).max()
			  << std::endl;

	std::cout << "Armadillo convolution: " << std::flush;
	auto result_arma_conv = ArmadilloConv(sig, ir);
	std::cout << "\n\tduration: " << result_arma_conv.first.count();
	std::cout << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_conv.second).max()
			  << std::endl;
}
