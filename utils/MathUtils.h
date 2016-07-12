//
// Created by alexwang on 7/12/16.
//

#ifndef BP_ANN_MATHUTILS_H
#define BP_ANN_MATHUTILS_H

#include <cstdlib>

namespace alex {
  class MathUtils {
  public:
    static double rand1() {
      return (2.0 * std::rand() / RAND_MAX) - 1;
    }

  };
}


#endif //BP_ANN_MATHUTILS_H
