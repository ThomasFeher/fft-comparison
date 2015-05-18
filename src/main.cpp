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
                           ,std::vector<AudioVec>& // input vector
						   ,AudioVec& // real vector
						   ,AudioVec& // imag vector
						   ,AudioVec&>; // output vector

namespace {
	uint32_t num = 1;
	uint32_t fft_size = 48000;
	uint32_t conv_ir_size = 72000;
	uint32_t conv_sig_size = 5760000;
	uint32_t min_block_size = 1024;
	//uint32_t conv_sig_size = 1024;
	// when false, the optimal blocksize and the corresponding number of
	// FFTs is calculated automatically 
	bool do_in_one_block = true;
}

duration<double> Convolution(std::vector<arma::fvec>& sig, arma::fvec ir, size_t fft_size, std::vector<arma::fvec> output) {
	ir = arma::flipud(ir);

	auto begin = high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<sig.size();++cnt) {
		arma::fvec sig_new = arma::zeros<arma::fvec>(sig[0].size() + 2*(ir.size()-1));
		sig_new.subvec(ir.size()-1, sig[cnt].size()+ir.size()-2) = sig[cnt];
		for (uint32_t sample_cnt=0;sample_cnt<fft_size;++sample_cnt) {
			output[cnt][sample_cnt] = 0;
			for (uint32_t ir_cnt=0;ir_cnt<::conv_ir_size;++ir_cnt) {
				output[cnt][sample_cnt] += sig_new[sample_cnt+ir_cnt] * ir[ir_cnt];
			}
		}
	}
	auto end = high_resolution_clock::now();

	return end - begin;
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

//duration<double> FftwConv(FftwData sig, FftwData ir) {
duration<double> FftwConv(AudioFFT sig_fft, const std::vector<AudioVec>& sig_in,AudioVec& sig_re, AudioVec& sig_im, AudioFFT ir_fft, const AudioVec& ir_in, AudioVec& ir_re, AudioVec& ir_im, std::vector<AudioVec>& output) {
	AudioVec re (sig_re.size());
	AudioVec im (sig_re.size());

	auto begin = high_resolution_clock::now();
	ir_fft.fft(ir_in.data(), ir_re.data(), ir_im.data());
	for (uint32_t block_cnt=0;block_cnt<sig_in.size();++block_cnt) {
		sig_fft.fft(sig_in[block_cnt].data(), sig_re.data(), sig_im.data());
		for (size_t cnt=0;cnt<sig_re.size();++cnt) {
			re[cnt] = sig_re[cnt]*ir_re[cnt] - sig_im[cnt]*ir_im[cnt];
			im[cnt] = sig_re[cnt]*ir_im[cnt] + sig_im[cnt]*ir_re[cnt];
		}
		sig_fft.ifft(output[block_cnt].data(), re.data(), im.data());
	}
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

std::tuple<size_t,size_t,uint32_t> CalcSizes(uint32_t ir_size, uint32_t sig_size, uint32_t min_block_size) {
	size_t sig_block_size = 0;
	while (sig_block_size < min_block_size) {
		size_t fft_size = pow(2,ceil(log2(ir_size)));
		sig_block_size = fft_size - ir_size + 1;
	}

	uint32_t block_num = sig_size / sig_block_size;

	return std::make_tuple(sig_block_size, fft_size, block_num);
}

std::tuple<std::vector<arma::fvec>,std::vector<AudioVec>,std::vector<arma::fvec>,std::vector<AudioVec>> PrepareData(const arma::fvec& sig, size_t sig_block_size, size_t ir_size) {
	std::vector<arma::fvec> sig_vec_arma;
	std::vector<arma::fvec> out_vec_arma;
	std::vector<AudioVec> sig_vec;
	std::vector<AudioVec> out_vec;

	for (uint32_t cnt=0;cnt<sig.size();cnt+=sig_block_size) {
		size_t end_idx = cnt+sig_block_size-1;
		if (end_idx > sig.size() - 1) {
			// last block with zeros appended
			end_idx = sig.size() -1;
			// correct sized block with zeros
			sig_vec_arma.push_back(arma::zeros<arma::fvec>(sig_block_size));
			sig_vec.push_back(std::vector<float> (sig_block_size,0));
			// copy last elements of signal to first elements of the last block
			sig_vec_arma[cnt].subvec(0,sig.size()-cnt+1) = sig(cnt,sig.size()-1);
			std::copy(sig.begin()+cnt, sig.end(), sig_vec[cnt].begin());
		}
		else {
			sig_vec_arma.push_back(sig.subvec(cnt,end_idx));
			AudioVec block;
			block.insert(block.end(), sig.begin()+cnt, sig.begin()+end_idx);
			sig_vec.push_back(block);
		}
		out_vec_arma.push_back(arma::zeros<arma::fvec>(sig_block_size+ir_size-1));
		out_vec.push_back(AudioVec(sig_block_size+ir_size-1,0));
	}
	
	return std::make_tuple(sig_vec_arma,sig_vec,out_vec_arma,out_vec);
}

int main(int argc, char* argv[]) {

	std::cout << "generating input signals" << std::endl;
	arma::fvec sig {arma::randn<arma::fvec>(::conv_sig_size)};
	arma::fvec ir {arma::randn<arma::fvec>(::conv_ir_size)};
	size_t out_size = ::conv_sig_size + ::conv_ir_size - 1;

	size_t sig_block_size;
	size_t fft_size;
	uint32_t block_num;
	std::tie(sig_block_size,fft_size,block_num) = CalcSizes(::conv_ir_size,
			                                                ::conv_sig_size,
															::min_block_size);
	std::vector<arma::fvec> sig_vec_arma;
	std::vector<arma::fvec> out_vec_arma;
	std::vector<AudioVec> sig_vec;
	std::vector<AudioVec> out_vec;
	std::tie(sig_vec_arma,sig_vec,out_vec_arma,out_vec) = PrepareData(sig,sig_block_size,::conv_ir_size);

	std::cout << "initializing FFTW" << std::flush;
	//AudioVec sig_vec  (fft_size, 0);
	//std::copy_n(sig.begin(), ::conv_sig_size, sig_vec.begin());
	AudioVec ir_vec (fft_size, 0);
	std::copy_n(ir.begin(), ::conv_ir_size, ir_vec.begin());
	AudioVec sig_re(AudioFFT::ComplexSize(fft_size)); 
	AudioVec ir_re(AudioFFT::ComplexSize(fft_size));
	AudioVec sig_im(AudioFFT::ComplexSize(fft_size)); 
	AudioVec ir_im(AudioFFT::ComplexSize(fft_size)); 
	AudioVec output(fft_size);
	AudioFFT fft_sig;
	// mearure only one initialization, for the second one we could just use
	// the wisdom of the first one.
	auto begin = high_resolution_clock::now();
	fft_sig.init(fft_size);
	auto end = high_resolution_clock::now();
	std::cout << "\n\tduration: " << (std::chrono::duration_cast<duration<double>>(end - begin)).count() << std::endl;
	AudioFFT fft_ir;
	fft_ir.init(fft_size);
	//FftwData sig_data = std::forward_as_tuple(fft_sig, sig_vec, sig_re, sig_im, output);
	//FftwData ir_data = std::forward_as_tuple(fft_ir, ir_vec, ir_re, ir_im, output);

	std::cout << "Armadillo FFT: " << std::flush;
	auto result_arma_fft = ArmadilloFft(sig);
	std::cout << "\n\tduration: " << result_arma_fft.first.count() << std::endl;

	// normal convolution, this is our reference output
	std::cout << "convolution: " << std::flush;
	auto result_conv = Convolution(sig_vec_arma, ir, fft_size, out_vec_arma);
	std::cout << "\n\tduration: " << result_conv.count() << std::endl;

	std::cout << "Armadillo FFT-Pow2-convolution: " << std::flush;
	auto result_arma_fft_pow2_conv = ArmadilloFftPow2Conv(sig, ir);
	std::cout << "\n\tduration: " << result_arma_fft_pow2_conv.first.count();
	std::cout << "\n\tmaximum difference of result: "
			  << arma::abs(result_conv.second - result_arma_fft_pow2_conv.second).max()
			  << std::endl;

	std::cout << "FFTW FFT-Pow2-convolution: " << std::flush;
	auto result_fftw_pow2_conv = FftwConv(fft_sig, sig_vec, sig_re, sig_im, fft_ir, ir_vec, ir_re, ir_im, output);
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
