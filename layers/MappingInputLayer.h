//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_MAPPINGINPUTLAYER_H
#define BP_ANN_MAPPINGINPUTLAYER_H


#include "InputLayer.h"
#include <functional>
#include <vector>

namespace alex {
class MappingInputLayer :public InputLayer {
public:
  MappingInputLayer(int InputWidth, std::vector<std::function<double(const double*, int)>> mappings)
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
  std::vector<std::function<double(const double *, int)>> mappings;
};

}

#endif //BP_ANN_MAPPINGINPUTLAYER_H
