#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <unistd.h>
#include <functional>

using namespace std;

const double LearningRate = 0.03;

double rand1() {
  return (2.0 * rand() / RAND_MAX) - 1;
}

class Layer {
public:
  Layer(int OutputWidth, Layer *last, int InputWidth = -1) :InputWidth(InputWidth), OutputWidth(OutputWidth), nextLayer(nullptr) {
    lastLayer = last;
    if (lastLayer) {
      lastLayer->nextLayer = this;
      this->InputWidth = lastLayer->InputWidth;
    }
    else if (InputWidth < 0) {
      throw std::invalid_argument("input width error");
    }
    lastInput = new double[this->InputWidth];
    lastOutput = new double[this->OutputWidth];
  }
  virtual ~Layer() {
    delete [] lastInput;
    delete [] lastOutput;
  }

  virtual void forwardPropagation(const double *input, double *output) const {
    for (int i = 0; i < InputWidth; ++i)
      lastInput[i] = input[i];
    for (int i = 0; i < OutputWidth; ++i) {
      lastOutput[i] = output[i];
    }
  }
  virtual void backwardPropagation(double *input, const double *output) = 0;

  Layer *getNextLayer() const {
    return nextLayer;
  }

  int InputWidth, OutputWidth;
protected:
  Layer *lastLayer, *nextLayer;

  double *lastInput;
  double *lastOutput;
};


class InputLayer :public Layer {
public:
  explicit InputLayer(int InputWidth) :Layer(InputWidth, nullptr, InputWidth) { }
  InputLayer(int InputWidth, int OutputWidth) :Layer(OutputWidth, nullptr, InputWidth) { }

  virtual void forwardPropagation(const double *input, double *output) const override {
    for (int i = 0; i < InputWidth; ++i) {
      output[i] = input[i];
    }
  }
  virtual void backwardPropagation(double *input, const double *output) override { }
};

class MappingInputLayer :public InputLayer {
public:
  MappingInputLayer(int InputWidth, vector<function<double(const double*, int)>> mappings)
      :InputLayer(InputWidth, InputWidth + (int)mappings.size()), mappings(mappings) {
  }

  virtual void forwardPropagation(const double *input, double *output) const override {
    for (int i = 0; i < InputWidth; ++i) {
      output[i] = input[i];
    }
    int i = InputWidth;
    for (auto mapping : mappings) {
      output[i] = mapping(input, InputWidth);
      i++;
    }
    Layer::forwardPropagation(input, output);
  }

private:
  vector<function<double(const double *, int)>> mappings;
};

class FullyConnectedLayer :public Layer {
public:
  FullyConnectedLayer(int OutputWidth, Layer *last) :Layer(OutputWidth, last) {
    weights = new double*[OutputWidth];
    bias = new double[OutputWidth];

    for (int i = 0; i < OutputWidth; ++i) {
      weights[i] = new double[InputWidth];
      for (int j = 0; j < InputWidth; ++j) {
        weights[i][j] = rand1();
      }
      bias[i] = rand1();
    }
  }
  virtual ~FullyConnectedLayer() override {
    for (int i = 0; i < OutputWidth; ++i) {
      delete [] weights[i];
    }
    delete [] weights;
    delete [] bias;
  }

  virtual void forwardPropagation(const double *input, double *output) const override {
    for (int i = 0; i < OutputWidth; ++i) {
      double sum = 0;
      for (int j = 0; j < InputWidth; ++j) {
        sum += input[j] * this->weights[i][j] + this->bias[i];
      }
      output[i] = tanh(sum);
    }
    Layer::forwardPropagation(input, output);
  }

  virtual void backwardPropagation(double *deltaKplus1 /* size: InputWidth */, const double *deltaKplus1Next) override {
    for (int j = 0; j < nextLayer->InputWidth; ++j) {
      deltaKplus1[j] = 0;
      for (int i = 0; i < nextLayer->OutputWidth; ++i) {
        deltaKplus1[j] += static_cast<FullyConnectedLayer*>(nextLayer)->getWeight(j, i) * deltaKplus1Next[i];
      }
      deltaKplus1[j] *= (1 - lastOutput[j] * lastOutput[j]);
    }

    for (int i = 0; i < OutputWidth; ++i) {
      for (int j = 0; j < InputWidth; ++j) {
        weights[i][j] -= LearningRate * deltaKplus1[i] * this->lastInput[j];
      }
      bias[i] -= LearningRate * deltaKplus1[i];
    }
  }

  double getWeight(int i, int o) {
    if (i < 0 || i >= InputWidth || o < 0 || o >= OutputWidth)
      throw std::out_of_range("getWeigth parameter out of range");
    return weights[o][i];
  }

  void printWeights() const {
    for (int i = 0; i < OutputWidth; ++i) {
      cout << '\t';
      for (int j = 0; j < InputWidth; ++j) {
        printf("%+03f ", weights[i][j]);
      }
      cout << endl;
    }
  }
protected:
  double **weights;
  double *bias;
};

class OutputLayer :public FullyConnectedLayer {
public:
  OutputLayer(int OutputWidth, Layer *last) :FullyConnectedLayer(OutputWidth, last) { }

  virtual void backwardPropagation(double *deltaKplus1, const double *error) override {
    for (int i = 0; i < OutputWidth; ++i) {
      deltaKplus1[i] = error[i] * (1 - this->lastOutput[i] * this->lastOutput[i]);
    }
    for (int i = 0; i < OutputWidth; ++i) {
      for (int j = 0; j < InputWidth; ++j) {
        weights[i][j] -= LearningRate * deltaKplus1[i] * this->lastInput[j];
      }
      bias[i] -= LearningRate * deltaKplus1[i];
    }
  }
};

void calculateError(const double *output, const double *expected, double *error, int width) {
  for (int i = 0; i < width; ++i) {
    error[i] = output[i] - expected[i];
  }
}

double absError(const double *err, int width) {
  double sum = 0;
  for (int i = 0; i < width; ++i) {
    sum += err[i] * err[i];
  }
  return sum;
}

const int InputWidth = 2;
const int OutputWidth = 3;

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

int realFunction(double *input, int inputCount, double *output, int outputCount) {
  double x = input[0], y = input[1];
  if ((x+2)*(x+2) + y*y < 1) {
    return setIndex(0, output, outputCount);
  }
  else if ((x-2)*(x-2) + y*y < 1) {
    return setIndex(1, output, outputCount);
  }
  else {
    return setIndex(2, output, outputCount);
  }
}
//int realFunction(double *input, int inputCount, double *output, int outputCount) {
//  if (inputCount != 2) {
//    throw std::invalid_argument("realFunction input not 2 dimension!");
//  }
//  double x = input[0], y = input[1];
//  if (2 * x + y < 0) {
//    return setIndex(0, output, outputCount);
//  }
//  else {
//    return setIndex(1, output, outputCount);
//  }
//}

std::vector<Layer*> initializeNetwork() {

  MappingInputLayer *inputLayer = new MappingInputLayer(InputWidth, {
      [](const double *input, int w) -> double { return input[0] * input[0]; },
      [](const double *input, int w) -> double { return input[1] * input[1]; }
  });
  FullyConnectedLayer *layer1 = new FullyConnectedLayer(3, inputLayer);
  FullyConnectedLayer *layer2 = new FullyConnectedLayer(5, layer1);
  OutputLayer *outputLayer = new OutputLayer(OutputWidth, layer2);

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

void trainAndTest(std::vector<Layer*>& layers, unsigned int trainTimes, unsigned int testTimes) {
  for (int j = 0; j < trainTimes; ++j) {
    double expectedOutput[OutputWidth], *inputData = new double[InputWidth], *outputData = nullptr;
    inputData[0] = rand1() * InputXWidth;
    inputData[1] = rand1() * InputYWidth;
    realFunction(inputData, InputWidth, expectedOutput, OutputWidth);

    for (auto layer : layers) {
      outputData = new double[layer->OutputWidth];
      layer->forwardPropagation(inputData, outputData);

      delete [] inputData;
      inputData = outputData;
    }

    calculateError(outputData, expectedOutput, outputData, OutputWidth);
    cout << "error   \t" << absError(outputData, OutputWidth) << endl;


    for (auto it = layers.rbegin(); it < layers.rend(); ++it) {
      auto layer = *it;
      inputData = new double[layer->OutputWidth];
      layer->backwardPropagation(inputData, outputData);

      delete [] outputData;
      outputData = inputData;
    }
  }

  int successCount = 0;
  for (int i = 0; i < testTimes; ++i) {
    double expectedOutput[OutputWidth], *inputData = new double[InputWidth], *outputData = nullptr;
    inputData[0] = rand1() * InputXWidth;
    inputData[1] = rand1() * InputYWidth;
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
  for (int i = 1; i < layers.size() - 1; ++i) {
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
        std::cout << "Basic Command Line Parameter App" << std::endl
        << desc << std::endl;
        return 0;
      }

      boost::program_options::notify(vm); // throws on error, so do after help in case
      // there are any problems
    }
    catch(boost::program_options::error& e)
    {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return 1;
    }

    if (vm.count("train-times"))
      trainTimes = vm["train-times"].as<unsigned>();
    if (vm.count("test-times"))
      testTimes = vm["test-times"].as<unsigned>();

    auto layers = initializeNetwork();
    trainAndTest(layers, trainTimes, testTimes);
  }
  catch(std::exception& e) {
    std::cerr << "Unhandled Exception reached the top of main: "
      << e.what() << ", application will now exit" << std::endl;
    return 2;
  }
  return 0;
}
