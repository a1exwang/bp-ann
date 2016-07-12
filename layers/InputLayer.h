//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_INPUTLAYER_H
#define BP_ANN_INPUTLAYER_H

#include "Layer.h"

namespace alex {
  class InputLayer :public Layer {
  public:
    explicit InputLayer(int InputWidth) :Layer(InputWidth, nullptr, InputWidth) { }
    InputLayer(int InputWidth, int OutputWidth) :Layer(OutputWidth, nullptr, InputWidth) { }

    virtual void forwardPropagation(const double *input, double *output) const override {
      for (int i = 0; i < InputWidth; ++i) {
        output[i] = input[i];
      }
    }
    virtual void backwardPropagation(double *input, const double *output, double learningRate) override { }
  };
}


#endif //BP_ANN_INPUTLAYER_H
