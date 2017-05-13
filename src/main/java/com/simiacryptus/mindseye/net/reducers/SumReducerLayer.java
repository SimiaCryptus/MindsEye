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

package com.simiacryptus.mindseye.net.reducers;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class SumReducerLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = -5171545060770814729L;
  
  public SumReducerLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    double outputA = IntStream.range(0, inObj[0].data.length).mapToDouble(dataIndex -> {
      double sum = 0;
      for (final NNResult element : inObj) {
        final double[] input = element.data[dataIndex].getData();
        for (final double element2 : input) {
          sum += element2;
        }
      }
      return sum;
    }).sum();
    return new NNResult(new Tensor(new int[]{1}, new double[]{outputA})) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            Tensor[] passbackA = IntStream.range(0, in_l.data.length).mapToObj(dataIndex -> {
              final double delta = data[0].get(0);
              final Tensor passback = new Tensor(in_l.data[dataIndex].getDims());
              for (int i = 0; i < in_l.data[dataIndex].dim(); i++) {
                passback.set(i, delta);
              }
              return passback;
            }).toArray(i -> new Tensor[i]);
            in_l.accumulate(buffer, passbackA);
          }
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive())
            return true;
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
