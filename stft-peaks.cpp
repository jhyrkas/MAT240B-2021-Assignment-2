// Assignment 2
//
// XXX Analysis!
// - find the most prominent N peaks in a sequence of sound clips
// - break the input into a sequence of overlaping sound clips
//   + each sound clip is 2048 samples long
//   + each sound clip should overlap the last by 1024 samples (50% overlap)
//   + use a Hann window to prepare the data before FFT analysis
// - use the FFT to analyse each sound clip
//   + use an FFT size of 8192 (pad the empty data with zeros)
//   + find the frequency and amplitude of each peak in the clip
//     * find maxima, calculate frequency, sort these by amplitude
//   + store the frequency and amplitude of the most prominent N of these
// - test this whole process using several given audio files (e.g.,
// sine-sweep.wav, impulse-sweep.wav, sawtooth-sweep.wav)
//
// -- Karl Yerkes / 2021-01-12 / MAT240B
//

#include <algorithm>  // std::sort
#include <cmath>
#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

// utility functions

// used in fft
// adapted from: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

typedef std::pair<double,double> amp_and_freq;
bool comparator ( const amp_and_freq& l, const amp_and_freq& r)
   { return l.first > r.first; } // sort descending

// higher memory implementation via http://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
 
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];
 
    // conquer
    fft(even);
    fft(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

// fixed size for now
double* hann_window() {
    static double window[2048];

    for (int i = 0; i < 2048; i++) {
        window[i] = 0.5 * (1.0 - cos(2.0*M_PI*(i+1)/2049.0));
    }

    return window;
}

int main(int argc, char *argv[]) {
  // take the data in
  //
  std::vector<double> data;
  double value;
  int n = 0;
  while (std::cin >> value) {
    data.push_back(value);
    n++;
  }

  // put your code here!
  //
  // for each sound clip of 2048 samples print information on the the N peaks.
  // print the N frequency/amplitude pairs on a single line. separate the
  // elements of each pair using the slash "/" character and separate each pair
  // using comma "," character. For example:
  //
  // 12.005/0.707,24.01/0.51,48.02/0.3 ...
  // 13.006/0.706,26.01/0.50,52.02/0.29 ...
  //
  // take N as an argument on the command line.
  //

  assert(argc > 1);
  int N = std::stof(argv[1]); // fix this later

  double* window = hann_window();
  int SAMPLING_RATE = 48000; // import?
  int hop_size = 1024;
  int nfft = 8192;
  int window_size = 2048;

  int nframes = ceil(n / float(hop_size));

  CArray fft_buf(nfft);
  int start_index = 0;

  for (int fr = 0; fr < nframes; fr++) {
     // PART 1: create fft buffer

     // should deal with size corner cases
     int end_index = std::min(n, start_index+hop_size);

     int j = 0;
     for (int i = start_index; i < end_index; i++) {
        fft_buf[j] = data[i];
        j++;
     }

     // zero-pad what's left
     while (j < nfft) {
        fft_buf[j] = 0.0;
        j++;
     }

     // PART 2: perform FFT

     // apply hann window
     for (int i = 0; i < window_size; i++) {
        fft_buf[i] *= window[i];
     }

     // can't remember if this is the way to do this
     fft(fft_buf);

     // PART 3: find peaks
     double bin_step = double(SAMPLING_RATE) / nfft;
     amp_and_freq spectrogram[nfft/2+1];
     // don't bother with negative frequencies
     for (int j = 0; j < nfft/2+1; j++) {
        spectrogram[j] = std::make_pair(std::abs(fft_buf[j]), j * bin_step);
     }

     std::sort(spectrogram, spectrogram + (nfft/2) + 1, comparator);
     for (int i = 0; i < N; i++) {
       std::cout << spectrogram[i].second << "/" << spectrogram[i].first;
       if (i < N-1) {
         std::cout << ",";
       } else {
         std::cout << std::endl;
       }
     }
     
     // next frame
     start_index += hop_size;
  }


  return 0;
}
