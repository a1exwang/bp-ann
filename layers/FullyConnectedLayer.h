//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_FULLYCONNECTEDLAYER_H
#define BP_ANN_FULLYCONNECTEDLAYER_H

#include "Layer.h"
#include "../utils/MathUtils.h"
#include <iostream>
#include <cmath>

namespace alex {
  class FullyConnectedLayer :public Layer {
  public:
    FullyConnectedLayer(int OutputWidth, Layer *last) :Layer(OutputWidth, last) {
      weights = new double*[OutputWidth];
      bias = new double[OutputWidth];

      for (int i = 0; i < OutputWidth; ++i) {
        weights[i] = new double[InputWidth];
        for (int j = 0; j < InputWidth; ++j) {
          weights[i][j] = MathUtils::rand1();
        }
        bias[i] = MathUtils::rand1();
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
        output[i] = std::tanh(sum);
      }
      Layer::forwardPropagation(input, output);
    }

    virtual void backwardPropagation(double *deltaKplus1, const double *deltaKplus1Next, double learningRate) override {
      for (int j = 0; j < nextLayer->InputWidth; ++j) {
        deltaKplus1[j] = 0;
        for (int i = 0; i < nextLayer->OutputWidth; ++i) {
          deltaKplus1[j] += static_cast<FullyConnectedLayer*>(nextLayer)->getWeight(j, i) * deltaKplus1Next[i];
        }
        deltaKplus1[j] *= (1 - lastOutput[j] * lastOutput[j]);
      }

      for (int i = 0; i < OutputWidth; ++i) {
        for (int j = 0; j < InputWidth; ++j) {
          weights[i][j] -= learningRate * deltaKplus1[i] * this->lastInput[j];
        }
        bias[i] -= learningRate * deltaKplus1[i];
      }
    }

    double getWeight(int i, int o) {
      if (i < 0 || i >= InputWidth || o < 0 || o >= OutputWidth)
        throw std::out_of_range("getWeigth parameter out of range");
      return weights[o][i];
    }

    void printWeights() const {
      for (int i = 0; i < OutputWidth; ++i) {
        std::cout << '\t';
        for (int j = 0; j < InputWidth; ++j) {
          printf("%+03f ", weights[i][j]);
        }
        std::cout << std::endl;
      }
    }
  protected:
    double **weights;
    double *bias;
  };
}

#endif //BP_ANN_FULLYCONNECTEDLAYER_H
