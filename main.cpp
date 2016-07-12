#include "layers/InputLayer.h"
#include "layers/OutputLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/MappingInputLayer.h"
#include "utils/MathUtils.h"
#include "train/SimpleTrainer.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <unistd.h>
#include <functional>

using namespace std;
using namespace alex;

const double LearningRate = 0.03;
const int InputWidth = 2;
const int OutputWidth = 2;

double InputXWidth = 3.0;
double InputYWidth = 1.0;

static int setIndex(int index, double *output, int outputCount) {
  if (index < 0 || index >= outputCount)
    throw new std::invalid_argument("setIndex argument out of bound");
  for (int i = 0; i < outputCount; ++i) {
    output[i] = -1;
    if (i == index)
      output[i] = 1;
  }
  return index;
}
auto realFunction = [](const double *input, int iw, double *output, int ow) -> int {
  double x = input[0], y = input[1];
  if ((x+2)*(x+2) + y*y < 1) {
    return setIndex(0, output, ow);
  }
  else {
    return setIndex(1, output, ow);
  }
};

std::vector<Layer*> initializeNetwork() {

  MappingInputLayer *inputLayer = new MappingInputLayer(InputWidth, {
      [](const double *input, int w) -> double { return sin(input[0]); },
      [](const double *input, int w) -> double { return sin(input[1]); },
      [](const double *input, int w) -> double { return input[0] * input[0]; },
      [](const double *input, int w) -> double { return input[0] * input[1]; },
//      [](const double *input, int w) -> double { return input[1] * input[1]; }
  });
  FullyConnectedLayer *layer1 = new FullyConnectedLayer(6, inputLayer);
  FullyConnectedLayer *layer3 = new FullyConnectedLayer(6, layer1);
  OutputLayer *outputLayer = new OutputLayer(OutputWidth, layer3);

  vector<Layer*> layers;
  Layer *last = inputLayer;
  layers.push_back(last);
  while (true) {
    Layer *current = last->getNextLayer();
    if (!current)
      break;
    layers.push_back(current);
    last = current;
  }

  return layers;
}

void testLayers(std::vector<Layer *> &layers, unsigned int testTimes) {
  int successCount = 0;
  for (int i = 0; i < testTimes; ++i) {
    double expectedOutput[OutputWidth], *inputData = new double[InputWidth], *outputData = nullptr;
    inputData[0] = MathUtils::rand1() * InputXWidth;
    inputData[1] = MathUtils::rand1() * InputYWidth;
    int expectedIndex = realFunction(inputData, InputWidth, expectedOutput, OutputWidth);

    for (auto layer : layers) {
      outputData = new double[layer->OutputWidth];
      layer->forwardPropagation(inputData, outputData);

      delete [] inputData;
      inputData = outputData;
    }

    double max = -1; int maxIndex = 0;
    for (int j = 0; j < OutputWidth; ++j) {
      if (inputData[j] > max) {
        max = inputData[j];
        maxIndex = j;
      }
    }
    if (maxIndex == expectedIndex)
      successCount++;

    delete [] outputData;
  }

  // NOTE: layer 0 is not a FullyConnectedLayer
  for (int i = 1; i < layers.size(); ++i) {
    cout << "layer" << i << " weights: " << endl;
    static_cast<FullyConnectedLayer*>(layers[i])->printWeights();
  }

  cout << "success rate = " << 100.0 * successCount / testTimes << "%" << endl;
}

#include <boost/program_options.hpp>
int main(int argc, const char *argv[]) {
  srand(1);

  unsigned testTimes = 10000,
          trainTimes = 10000;

  try {
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("help", boost::program_options::value<string>(), "Print help messages")
            ("train-times", boost::program_options::value<unsigned int>(), "train times")
            ("test-times", boost::program_options::value<unsigned int>(), "test times");

    boost::program_options::variables_map vm;
    try {
      boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc),
                vm); // can throw

      /** --help option
       */
      if (vm.count("help")) {
        std::cout << "bp-ann" << std::endl
        << desc << std::endl;
        return 0;
      }

      boost::program_options::notify(vm); // throws on error, so do after help in case
      // there are any problems
    }
    catch(boost::program_options::error& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return 1;
    }

    if (vm.count("train-times"))
      trainTimes = vm["train-times"].as<unsigned>();
    if (vm.count("test-times"))
      testTimes = vm["test-times"].as<unsigned>();

    auto layers = initializeNetwork();
    SimpleTrainer trainer(InputWidth, OutputWidth, LearningRate, trainTimes, layers,
                          [=](double *input, int w) -> void {
                            input[0] = MathUtils::rand1() * InputXWidth;
                            input[1] = MathUtils::rand1() * InputYWidth;
                          },
                          realFunction,
                          [](const double *output, const double *expected, double *error, int width) -> void {
                            for (int i = 0; i < width; ++i) {
                              error[i] = output[i] - expected[i];
                            }
                          },
                          [](const double *err, int width) -> double {
                            double sum = 0;
                            for (int i = 0; i < width; ++i) {
                              sum += err[i] * err[i];
                            }
                            return sum;
                          });
    trainer.startTraining();
    testLayers(layers, testTimes);
  }
  catch(std::exception& e) {
    std::cerr << "Unhandled Exception reached the top of main: "
      << e.what() << ", application will now exit" << std::endl;
    return 2;
  }
  return 0;
}
