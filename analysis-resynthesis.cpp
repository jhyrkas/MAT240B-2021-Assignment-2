// Assignment 2
//
// - make an Analysis-Resynthesis / Sinusoidal Modeling app
// - place this file (assgnment2.cpp) in a folder in allolib_playground/
// - build with ./run.sh folder/analysis-resynthesis.cpp
// - add your code to this file
//
// -- Karl Yerkes / 2021-01-12 / MAT240B
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>  // std::sort
#include <cmath>      // ::sin()
#include <complex>
#include <exception>
#include <iostream>
#include <valarray>
#include <vector>

#include "al/app/al_App.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"
#include "dr_wav.h"

// define free functions (functions not associated with any class) here
//

// a class that encapsulates data approximating one cycle of a sine wave
//
struct SineTable {
  std::vector<double> data;
  SineTable(int n = 16384) {
    data.resize(n);
    for (int i = 0; i < n; i++) {
      data[i] = ::sin(M_PI * 2.0 * i / n);
      // printf("%lf\n", data[i]);
    }
  }
};

// a function that works with the class above to return the value of a sine
// wave, given a value **in terms of normalized phase**. p should be on (0, 1).
//
double sine(double p) {
  static SineTable table;
  int n = table.data.size();
  int a = p * n;
  int b = 1 + a;
  double t = p * n - a;
  if (b == n)  //
    b = 0;
  // linear interpolation
  return (1 - t) * table.data[a] + t * table.data[b];
}

// a constant global "variable" as an alternative to a pre-processor definition
const double SAMPLE_RATE = 48000.0;
//#define SAMPLE_RATE (48000.0)  // pre-processor definition

// a class using the operator/functor pattern for audio synthesis/processing. a
// Phasor or "ramp" wave goes from 0 to 1 in a upward ramping sawtooth shape. it
// may be used as a phase value in other synths.
//
struct Phasor {
  double phase = 0;
  double increment = 0;
  void frequency(double hz) {  //
    increment = hz / SAMPLE_RATE;
  }
  double operator()() {
    double value = phase;
    phase += increment;
    if (phase >= 1)  //
      phase -= 1;
    return value;
  }
};

// a class that may be used as a Sine oscillator
//
struct Sine : Phasor {
  double operator()() {  //
    return sine(Phasor::operator()());
  }
};

// suggested entry in a table of data resulting from the analysis of the input
// sound.
struct Entry {
  double frequency, amplitude;
};

bool entry_comparator ( const Entry& l, const Entry& r)
   { return l.frequency < r.frequency; } // sort ascending

// from stft-peaks.cpp
// used in fft
// adapted from: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

typedef std::pair<double,double> amp_and_freq;
bool amp_freq_comparator ( const amp_and_freq& l, const amp_and_freq& r)
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

std::vector<std::vector<Entry>> stft_peaks(float* data, int data_length, int N) {
    std::vector<std::vector<Entry>> entries;

    double* window = hann_window();
    int hop_size = 1024;
    int nfft = 8192;
    int window_size = 2048;

    int nframes = ceil(data_length / float(hop_size));

    CArray fft_buf(nfft);
    int start_index = 0;

    for (int fr = 0; fr < nframes; fr++) {
        // PART 1: create fft buffer

        // should deal with size corner cases
        int end_index = std::min(data_length, start_index+hop_size);

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
        double bin_step = double(SAMPLE_RATE) / nfft;
        std::vector<amp_and_freq> peaks;
        // avoid giving peaks at DC or Nyquist
        // don't bother with negative frequencies
        for (int j = 1; j < nfft/2; j++) {
            double amp = std::abs(fft_buf[j]);
            // making one of these >= so that only one value in a plateau is captured
            if (amp > std::abs(fft_buf[j-1]) && amp >= std::abs(fft_buf[j+1])) {
                peaks.push_back(std::make_pair(std::abs(fft_buf[j]), j * bin_step));
            }
        }

        std::sort(peaks.begin(), peaks.end(), amp_freq_comparator);
        entries.push_back(std::vector<Entry>());
        for (int i = 0; i < N; i++) {
            // this will be 0 amp anyway, setting this high because of
            // how i am assigning voices (i.e. voice 0 always gets the
            // lowest frequency)
            double freq = SAMPLE_RATE / 2.0;
            double amp = 0.0;
            if (i < peaks.size()) {
                freq = peaks[i].second;
                amp = peaks[i].first;
            }
            Entry e = {freq, amp};
            entries[fr].push_back(e);
        }

        // re-sort so that entries are sorted low to high in frequency
        std::sort(entries[fr].begin(), entries[fr].end(), entry_comparator);

        // next frame
        start_index += hop_size;
    }

    return entries;
}

struct MyAppCreationException : public std::exception {
  virtual const char* what() const throw() {
    return "Couldn't open WAV file";
  }
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

using namespace al;

struct MyApp : App {
  Parameter background{"background", "", 0.0, "", 0.0f, 1.0f};
  Parameter t{"t", "", 0.0, "", 0.0f, 1.0f};
  ControlGUI gui;

  int N;  // the number of sine oscillators to use

  std::vector<Sine> sine;
  std::vector<std::vector<Entry>> data;

  MyApp(int argc, char *argv[]) {
    // C++ "constructor" called when MyApp is declared
    //

    // XXX Analysis!
    // - reuse your STFT peaks analysis code here
    // - fill the data structure (i.e., std::vector<std::vector<Entry>> data)
    // - accept and process command line arguments here
    //   + the name of a .wav file
    //   + the number of oscillators N
    // - adapt code from wav-read.cpp
    
    // wav-read.cpp
    drwav* pWav = drwav_open_file(argv[1]);
    if (pWav == nullptr) {
        throw MyAppCreationException();
    }

    float* pSampleData = (float*)malloc((size_t)pWav->totalPCMFrameCount *
                                      pWav->channels * sizeof(float));
    drwav_read_f32(pWav, pWav->totalPCMFrameCount, pSampleData);

    drwav_close(pWav);

    N = std::atoi(argv[2]);

    data = stft_peaks(pSampleData, pWav->totalPCMFrameCount, N);
    free(pSampleData);

    // need big time gain normalization here
    double max_amp = 0.0;
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < N; j++) {
            max_amp = std::max(max_amp, data[i][j].amplitude);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < N; j++) {
            data[i][j].amplitude /= max_amp;
        }
    }
  }

  void onInit() override {
    // called a single time just after the app is started
    //

    sine.resize(N);

    // remove this code later. it's just here to test
    for (int n = 0; n < N; n++) {
      sine[n].frequency(220.0 * (1 + n));
    }
  }

  void onCreate() override {
    // called a single time (in a graphics context) before onAnimate or onDraw
    //

    nav().pos(Vec3d(0, 0, 8));  // Set the camera to view the scene

    gui << background;
    gui << t;
    gui.init();

    // Disable nav control; So default keyboard and mouse control is disabled
    navControl().active(false);
  }

  void onAnimate(double dt) override {
    // called over and over just before onDraw
    t.set(t.get() + dt * 0.03);
    if (t > 1) {
      t.set(t.get() - 1);
    }
  }

  void onDraw(Graphics &g) override {
    // called over and over, once per view, per frame. warning! this may be
    // called more than once per frame. for instance, in the context of 3D
    // stereo viewing, this will be called twice per frame. if 6 screens are
    // attached to this system, then onDraw will be called 6 times per frame.
    //
    g.clear(background);
    //
    //

    // Draw th GUI
    gui.draw(g);
  }

  void onSound(AudioIOData &io) override {
    // called over and over, once per audio block
    //

    // XXX Resynthesis!
    // - add code here
    // - use data from the std::vector<std::vector<Entry>> to adjust the
    // frequency of the N oscillators
    // - use the value of the t parameter to determine which part of the sound
    // to resynthesize
    // - use linear interpolation
    //

    float t_val = t.get();
    float frac_ind = t_val * data.size();
    // need to check for both low and high because t can still be 1.0
    int low_ind = (int)frac_ind;
    int high_ind = low_ind + 1;
    // special checks as t approaches 1.0
    if (low_ind >= data.size()) {
        low_ind = data.size() - 1;
    }
    if (high_ind >= data.size()) {
        high_ind = data.size() - 1;
    }

    float upper_weight = frac_ind - low_ind;
    float lower_weight = 1.0 - upper_weight;
    /*
    if (low_ind < 0) {
        printf("oops1\n");
    }
    if (low_ind >= data.size()) {
        printf("oops2\n");
    }
    if (high_ind >= data.size()) {
        printf("oops3\n");
    }
    if (lower_weight < 0.0) {
        printf("oops4\n");
    }
    if (lower_weight > 1.0) {
        printf("oops5\n");
    }
    if (upper_weight < 0.0) {
        printf("oops6\n");
    }
    if (upper_weight > 1.0) {
        printf("oops7\n");
    }*/

    while (io()) {
      // add the next sample from each of the N oscillators
      //
      float f = 0;
      for (int n = 0; n < N; n++) {
        //float freq = data[low_ind][n].frequency;
        //float amp = data[low_ind][n].amplitude;

        float freq = (lower_weight * data[low_ind][n].frequency) + (upper_weight * data[high_ind][n].frequency);
        float amp = ((lower_weight * data[low_ind][n].amplitude) + (upper_weight * data[high_ind][n].amplitude));
        sine[n].frequency(freq);
        f += amp*sine[n]();  // XXX update this line to effect amplitude
      }
      f /= N;  // reduce the amplitude of the result
      io.out(0) = f;
      io.out(1) = f;
    }
  }

  bool onKeyDown(const Keyboard &k) override {
    int midiNote = asciiToMIDI(k.key());
    return true;
  }

  bool onKeyUp(const Keyboard &k) override {
    int midiNote = asciiToMIDI(k.key());
    return true;
  }
};

int main(int argc, char *argv[]) {
    // wav-read.cpp
    if (argc < 3) {
        printf("usage: analysis-resynthesis wav-file num-oscs");
        return 1;
    }

    // MyApp constructor called here, given arguments from the command line
    MyApp app(argc, argv);

    //app.configureAudio(48000, 512, 2, 1);
    // seems like i need to really decrease the audio rate to stop clicking
    app.configureAudio(22050, 512, 2, 1);

    // Start the AlloLib framework's "app" construct. This blocks until the app is
    // quit (or it crashes).
    app.start();

    return 0;
}
