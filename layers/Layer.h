//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_LAYER_H
#define BP_ANN_LAYER_H

#include <stdexcept>

namespace alex {
  class Layer {
  public:
    Layer(int OutputWidth, Layer *last, int InputWidth = -1) :InputWidth(InputWidth), OutputWidth(OutputWidth), nextLayer(nullptr) {
      lastLayer = last;
      if (lastLayer) {
        lastLayer->nextLayer = this;
        this->InputWidth = lastLayer->OutputWidth;
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
    virtual void backwardPropagation(double *input, const double *output, double learningRate) = 0;

    Layer *getNextLayer() const {
      return nextLayer;
    }

    int InputWidth, OutputWidth;
  protected:
    Layer *lastLayer, *nextLayer;

    double *lastInput;
    double *lastOutput;
  };
}


#endif //BP_ANN_LAYER_H
