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

package com.simiacryptus.mindseye.layers.loss;

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class EntropyLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = -6257785994031662519L;
  
  public EntropyLossLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor gradientA[] = new Tensor[inObj[0].data.length];
    Tensor[] outputA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
      final Tensor l = inObj[0].data[dataIndex];
      final Tensor b = inObj[1].data[dataIndex];
      final Tensor gradient = new Tensor(l.getDims());
      gradientA[dataIndex] = gradient;
      final double[] gradientData = gradient.getData();
      final double descriptiveNats;
      double total = 0;
      for (int i = 0; i < l.dim(); i++) {
        final double ad = Math.max(Math.min(l.getData()[i], 1.), 1e-12);
        final double bd = b.getData()[i];
        gradientData[i] = -bd / ad;
        total += -bd * Math.log(ad);
      }
      descriptiveNats = total;
      
      return new Tensor(new int[]{1}, new double[]{descriptiveNats});
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(gradientA[dataIndex].getDims());
            for (int i = 0; i < inObj[0].data[0].dim(); i++) {
              passback.set(i, data[dataIndex].get(0) * gradientA[dataIndex].get(i));
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, passbackA);
          }
          if (inObj[1].isAlive())
            throw new RuntimeException();
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
