/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.layers.activation.ActivationLayerTestBase;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public abstract class LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(ActivationLayerTestBase.class);
  
  @Test
  public void testDerivatives() throws Throwable {
    Tensor[] inputPrototype = Arrays.stream(getInputDims()).map(dim->new Tensor(dim).fill(() -> Util.R.get().nextDouble()))
                                .toArray(i->new Tensor[i]);
    Tensor outputPrototype = GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
      (data, exe) -> getLayer().eval(exe, NNResult.batchResultArray(data.toArray(new Tensor[][]{}))).getData().get(0),
      (a, b) -> a.add(b));
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
  }
  
  public DerivativeTester getDerivativeTester() {
    return new DerivativeTester(1e-4, 1e-8);
  }
  
  public abstract NNLayer getLayer();
  
  public abstract int[][] getInputDims();
}
