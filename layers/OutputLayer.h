//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_OUTPUTLAYER_H
#define BP_ANN_OUTPUTLAYER_H

#include "FullyConnectedLayer.h"

namespace alex {
  class OutputLayer :public FullyConnectedLayer {
  public:
    OutputLayer(int OutputWidth, Layer *last) :FullyConnectedLayer(OutputWidth, last) { }

    virtual void backwardPropagation(double *deltaKplus1, const double *error, double learningRate) override {
      for (int i = 0; i < OutputWidth; ++i) {
        deltaKplus1[i] = error[i] * (1 - this->lastOutput[i] * this->lastOutput[i]);
      }
      for (int i = 0; i < OutputWidth; ++i) {
        for (int j = 0; j < InputWidth; ++j) {
          weights[i][j] -= learningRate * deltaKplus1[i] * this->lastInput[j];
        }
        bias[i] -= learningRate * deltaKplus1[i];
      }
    }
  };
}


#endif //BP_ANN_OUTPUTLAYER_H
