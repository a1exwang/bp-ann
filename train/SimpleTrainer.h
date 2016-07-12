//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_SIMPLETRAIN_H
#define BP_ANN_SIMPLETRAIN_H
#include "../layers/Layer.h"
#include "../utils/MathUtils.h"

#include <vector>
#include <iostream>
#include <functional>

namespace alex {
  class SimpleTrainer {
  public:
    SimpleTrainer(int inputWidth, int outputWidth, double learningRate, int trainingTimes,
                  std::vector<Layer*> layers,
                  std::function<void(double*, int)> inputGenerator,
                  std::function<int(const double*, int, double*, int)> realFunction,
                  std::function<void(const double*, const double*, double*, int)> errorFunction,
                  std::function<double(const double*, int)> scalarError)
        :InputWidth(inputWidth), OutputWidth(outputWidth), LearningRate(learningRate),
         TrainingTimes(trainingTimes),
         layers(layers),
         inputGenerator(inputGenerator),
         realFunction(realFunction),
         errorFunction(errorFunction),
         scalarError(scalarError) { }

    void startTraining() {

      for (int j = 0; j < TrainingTimes; ++j) {
        double expectedOutput[OutputWidth],
            *inputData = new double[InputWidth],
            *outputData = nullptr;
        inputGenerator(inputData, InputWidth);
        realFunction(inputData, InputWidth, expectedOutput, OutputWidth);

        for (auto layer : layers) {
          outputData = new double[layer->OutputWidth];
          layer->forwardPropagation(inputData, outputData);

          delete [] inputData;
          inputData = outputData;
        }

        this->errorFunction(outputData, expectedOutput, outputData, OutputWidth);
        std::cout << "error   \t" << scalarError(outputData, OutputWidth) << std::endl;

        for (auto it = layers.rbegin(); it < layers.rend(); ++it) {
          auto layer = *it;
          inputData = new double[layer->OutputWidth];
          layer->backwardPropagation(inputData, outputData, LearningRate);

          delete [] outputData;
          outputData = inputData;
        }
      }
    }

    const std::vector<Layer*>& getLayers() const {
      return layers;
    }

  private:
    const int InputWidth, OutputWidth;
    const double LearningRate;
    const int TrainingTimes;
    std::vector<Layer*> layers;
    std::function<void(double*, int)> inputGenerator;
    std::function<int(const double*, int, double*, int)> realFunction;
    std::function<void(const double*, const double*, double*, int)> errorFunction;
    std::function<double(const double*, int)> scalarError;
  };


}
#endif //BP_ANN_SIMPLETRAIN_H
