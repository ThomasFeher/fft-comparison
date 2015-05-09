#include <chrono>
#include <iostream>

#include <armadillo>

namespace {
	uint32_t num = 1000000;
	uint32_t fft_size = 1024;
}

std::chrono::duration<double> armadillo(uint32_t fft_size) {
	arma::fvec input {arma::randn<arma::fvec>(fft_size)};
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
	for (uint32_t cnt=0;cnt<::num;++cnt) {
		auto output = arma::fft(input);
	}
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	return end - begin;
}

int main(int argc, char* argv[]) {
	auto arma_time = armadillo(::fft_size);

	std::cout << "armadillo: " << arma_time.count() << std::endl;
}
